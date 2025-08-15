import os
import numpy as np
from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
from mlir.passmanager import PassManager
from typing import Union, Optional
import multiprocessing
from rl_autoschedular import config as cfg
from rl_autoschedular.state import OperationState, BenchmarkFeatures
from utils.log import print_alert
from statistics import median
import json
import re


def evaluate_code(state: OperationState, bench_data: BenchmarkFeatures, tmp_exec_data_file: str) -> tuple[Optional[int], Union[Exception, bool]]:
    """Evaluates the given MLIR code with a timeout.

    Args:
        state (OperationState): The operation state to evaluate.
        bench_data (BenchmarkFeatures): The benchmark features data.
        tmp_exec_data_file (str): The path to the temporary execution data file.

    Returns:
        Optional[float]: the execution time in seconds.
        Union[Exception, bool]: the assertion result or an exception if an error occurred.
    """
    code_cache_key = get_code_cache_key(state, bench_data)
    cache_exec_time = __check_execution_cache(state.bench_name, code_cache_key, tmp_exec_data_file)
    if cache_exec_time is not None:
        return cache_exec_time, True
    print_alert('Cache miss')

    real_exec_time, success = evaluate_code_with_bindings(state.transformed_code)

    return real_exec_time, success


# ================================== Evaluation Functions (Python Bindings) ==================================

def evaluate_code_with_bindings(code: str) -> tuple[Optional[int], Union[Exception, bool]]:
    """Lowers and runs the given MLIR code using Python bindings, then returns the execution time and assertion
    result (if the executed code returns the correct result).

    Args:
        code (str): The MLIR code to run.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
    """
    pass_pipeline = """builtin.module(
        loop-invariant-code-motion,
        canonicalize,

        eliminate-empty-tensors,
        empty-tensor-to-alloc-tensor,
        one-shot-bufferize{
            bufferize-function-boundaries
            function-boundary-type-conversion=identity-layout-map
        },

        convert-linalg-to-loops,
        canonicalize,
        buffer-deallocation-pipeline,
        convert-bufferization-to-memref,
        scf-forall-to-parallel,
        convert-scf-to-openmp,
        expand-strided-metadata,
        finalize-memref-to-llvm,
        convert-scf-to-cf,
        lower-affine,

        convert-openmp-to-llvm,
        convert-vector-to-llvm,
        convert-math-to-llvm,
        finalize-memref-to-llvm,
        convert-func-to-llvm,
        convert-index-to-llvm,
        convert-arith-to-llvm,
        convert-cf-to-llvm,

        reconcile-unrealized-casts,
        canonicalize,
        cse
    )"""

    with Context():
        module = Module.parse(code)
        pm = PassManager.parse(pass_pipeline)
        pm.run(module.operation)
    execution_engine = ExecutionEngine(
        module,
        opt_level=3,
        shared_libs=os.getenv("MLIR_SHARED_LIBS", "").split(","),
    )

    inputs = __create_inputs(code)

    args = []
    for input_arg in inputs:
        args.append(ctypes.pointer(ctypes.pointer(
            get_ranked_memref_descriptor(input_arg)
        )))

    delta_arg = (ctypes.c_int64 * 1)(0)
    args.append(delta_arg)

    times = []
    try:
        for _ in range(5):
            execution_engine.invoke("main", *args)
            times.append(delta_arg[0])
    except Exception as e:
        return None, e

    return median(times), True


def evaluate_code_with_bindings_wrapper(code: str, exec_times, assertions):
    """Wrapper function for evaluate_code_with_bindings to be used in multiprocessing.

    Args:
        code (str): The MLIR code to run.
        function_name (str): The name of the function to run.
        exec_times (list): A list to store the execution times.
        assertions (list): A list to store the assertion results
    """
    exec_time, assertion = evaluate_code_with_bindings(code)
    exec_times.append(exec_time)
    assertions.append(assertion)


def evaluate_code_with_bindings_and_timeout(code: str, timeout: Optional[float]) -> tuple[Optional[int], Union[Exception, bool]]:
    """Evaluates the given MLIR code using Python bindings with a timeout.

    Args:
        code (str): The MLIR code to run.
        function_name (str): The name of the function to run.
        timeout (Optional[float]): The timeout in seconds.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
    """
    manager = multiprocessing.Manager()
    exec_times = manager.list()
    assertions = manager.list()
    process = multiprocessing.Process(target=evaluate_code_with_bindings_wrapper, args=(code, exec_times, assertions))
    process.start()
    process.join(timeout)

    if process.is_alive():
        # The function is still running, terminate the process
        process.terminate()
        process.join()

        return None, False
    else:
        # The function completed within the timeout
        return exec_times[0], assertions[0]


# ================================== Evaluation Functions (MLIR CPU Runner) ==================================

def evaluate_code_with_cmd(code: str, tmp_file_path: str) -> tuple[Optional[int], bool]:
    """Lowers and runs the given MLIR code using MLIR opt and MLIR CPU Runner, then returns the execution time and assertion.

    Args:
        code (str): The MLIR code to run.
        tmp_file_path (str): The temporary file path to write the MLIR code.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
    """
    command_1 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt -loop-invariant-code-motion -canonicalize -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' -convert-vector-to-scf -convert-linalg-to-loops -buffer-deallocation-pipeline -convert-bufferization-to-memref -scf-forall-to-parallel -convert-scf-to-openmp -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-openmp-to-llvm -convert-vector-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -convert-index-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm -reconcile-unrealized-casts -canonicalize -cse"
    command_2 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs={os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_c_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libomp.so"

    with open(tmp_file_path, "w") as file:
        file.write(code)

    out = os.popen(f"""{command_1} {tmp_file_path} | {command_2} /dev/stdin""").read()

    if out:
        return int(out.strip().split('\n')[-1]), True
    else:
        return None, False


def evaluate_code_with_cmd_wrapper(code: str, tmp_file_path: str, exec_times, assertions):
    """Wrapper function for evaluate_code_with_cmd to be used in multiprocessing.

    Args:
        code (str): The MLIR code to run.
        tmp_file_path (str): The temporary file path to write the MLIR code.
        exec_times (list): A list to store the execution times.
        assertions (list): A list to store the assertion results
    """
    exec_time, assertion = evaluate_code_with_cmd(code, tmp_file_path)
    exec_times.append(exec_time)
    assertions.append(assertion)


def evaluate_code_with_cmd_and_timeout(code: str, tmp_file_path: str, timeout: Optional[float]) -> tuple[Optional[int], bool]:
    """Evaluates the given MLIR code using MLIR opt and MLIR CPU Runner with a timeout.

    Args:
        code (str): The MLIR code to run.
        tmp_file_path (str): The temporary file path to write the MLIR code.
        timeout (Optional[float]): The timeout in seconds.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
    """
    manager = multiprocessing.Manager()
    exec_times = manager.list()
    assertions = manager.list()
    process = multiprocessing.Process(target=evaluate_code_with_cmd_wrapper, args=(code, tmp_file_path, exec_times, assertions))
    process.start()
    process.join(timeout)

    if process.is_alive():
        # The function is still running, terminate the process
        process.terminate()
        process.join()

        return None, False
    else:
        # The function completed within the timeout
        return exec_times[0], assertions[0]


def __check_execution_cache(bench_name: str, cache_key: str, tmp_exec_data_file: str) -> Optional[int]:
    """Check the execution cache for the given operation state.

    Args:
        bench_name (str): The benchmark name to check.
        cache_key (str): The cache key to check.
        tmp_exec_data_file (str): The path to the temporary execution data file.

    Returns:
        Optional[int]: the execution time in nanoseconds if the operation is found in the cache, otherwise None.
    """
    # Start by checking the main execution cache file
    if cfg.exec_data_file:
        try:
            with open(cfg.exec_data_file, "r") as file:
                data = json.load(file)

            if bench_name in data and cache_key in data[bench_name]:
                return int(data[bench_name][cache_key])
        except Exception:
            pass

    # If no hit in the main cache file, check the temporary cache file
    with open(tmp_exec_data_file, "r") as file:
        data = json.load(file)

    if bench_name in data and cache_key in data[bench_name]:
        return int(data[bench_name][cache_key])

    # No hit in both cache files
    return None


def update_execution_cache(bench_name: str, cache_key: str, exec_time: int, tmp_exec_data_file: str):
    """Update the temp execution cache with the given operation state.

    Args:
        bench_name (str): The benchmark name to update.
        cache_key (str): The cache key to update.
        exec_time (int): The execution time in nanoseconds.
        tmp_exec_data_file (str): The path to the temporary execution data file.

    """
    with open(tmp_exec_data_file, "r") as file:
        data = json.load(file)

    if bench_name not in data:
        data[bench_name] = {}

    if cache_key in data[bench_name]:
        print_alert("Unexpected hit", data[bench_name][cache_key], exec_time)
        return
    data[bench_name][cache_key] = exec_time

    with open(tmp_exec_data_file, "w") as file:
        json.dump(data, file, indent=4)


def bulk_update_execution_cache(new_data: dict[str, dict[str, int]], tmp_exec_data_file: str):
    """Bulk update the temp execution cache with the given operation states.

    Args:
        new_data (dict[str, dict[str, int]]): The new data to update.
        tmp_exec_data_file (str): The path to the temporary execution data file.
    """
    with open(tmp_exec_data_file, "r") as file:
        data = json.load(file)

    for bench_name, bench_data in new_data.items():
        if bench_name not in data:
            data[bench_name] = {}

        for cache_key, exec_time in bench_data.items():
            if cache_key in data[bench_name]:
                print_alert("Unexpected hit", data[bench_name][cache_key], exec_time)
                continue
            data[bench_name][cache_key] = exec_time

    with open(tmp_exec_data_file, "w") as file:
        json.dump(data, file, indent=4)


def get_code_cache_key(state: OperationState, bench_data: BenchmarkFeatures) -> str:
    """Get the code cache key for the given operation state.

    Args:
        state (OperationState): The operation state to get the code cache key.
        bench_data (BenchmarkFeatures): The benchmark features data.

    Returns:
        str: the code cache key.
    """
    ops_codes = [''] * len(bench_data.operation_tags)
    for i, seq in enumerate(reversed(state.transformation_history)):
        # TODO: There might be edge cases where part of a seq is invalid `env.py:125`
        ops_codes[i] = ''.join(map(str, seq))

    return '|'.join(reversed(ops_codes))


def __create_inputs(code) -> list[np.ndarray]:
    main_pattern = r"func.func @main\(([^)]+)\)"
    main_params = re.search(main_pattern, code).group(1)
    main_shapes = [arg.split(':')[1].strip() for arg in main_params.split(',')]

    inputs: list[np.ndarray] = []
    for shape in main_shapes:
        assert shape.startswith('memref<') or shape.startswith('tensor<'), f'unexpected shape {shape}'
        *np_shape, dtype = shape.replace('memref<', '').replace('tensor<', '').replace('>', '').split('x')
        assert dtype[0] in ['f', 'i'] and dtype[1:] in ['32', '64'], f'unexpected dtype {dtype}'
        match dtype[0]:
            case 'f':
                match dtype[1:]:
                    case '32':
                        np_dtype = np.float32
                    case '64':
                        np_dtype = np.float64
            case 'i':
                match dtype[1:]:
                    case '32':
                        np_dtype = np.int32
                    case '64':
                        np_dtype = np.int64
        np_shape = list(map(int, np_shape))
        # if len(np_shape) > 0:
        #     inputs.append((np.random.rand(*np_shape) * 100).astype(np_dtype))
        # else:
        #     inputs.append(np.array(np.random.rand() * 100, dtype=np_dtype))
        inputs.append(np.zeros(np_shape, dtype=np_dtype))

    return inputs
