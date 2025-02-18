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
import json


def evaluate_code_with_timeout(state: OperationState, bench_data: BenchmarkFeatures, timeout: Optional[float] = 120) -> tuple[Optional[int], Union[Exception, bool]]:
    """Evaluates the given MLIR code with a timeout.

    Args:
        state (OperationState): The operation state to evaluate.
        bench_data (BenchmarkFeatures): The benchmark features data.
        timeout (Optional[float]): The timeout in seconds.

    Returns:
        Optional[float]: the execution time in seconds.
        Union[Exception, bool]: the assertion result or an exception if an error occurred.
    """
    code_cache_key = __get_code_cache_key(state, bench_data)
    cache_exec_time = __check_execution_cache(state.bench_name, code_cache_key)
    if cache_exec_time is not None:
        return cache_exec_time, True
    print_alert('Cache miss')

    if cfg.use_bindings:
        real_exec_time, success = evaluate_code_with_bindings_and_timeout(state.transformed_code, state.bench_name, timeout)
    else:
        real_exec_time, success = evaluate_code_with_cmd_and_timeout(state.transformed_code, state.tmp_file, timeout)

    if success and real_exec_time is not None:
        __update_execution_cache(state.bench_name, code_cache_key, real_exec_time)

    return real_exec_time, success


# ================================== Evaluation Functions (Python Bindings) ==================================

# TODO: Adapt this function to be able to run code without benchmark name
def evaluate_code_with_bindings(code: str, function_name: str) -> tuple[Optional[int], Union[Exception, bool]]:
    """Lowers and runs the given MLIR code using Python bindings, then returns the execution time and assertion
    result (if the executed code returns the correct result).

    Args:
        code (str): The MLIR code to run.
        function_name (str): The name of the function to run.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
    """
    pass_pipeline = """builtin.module(
        loop-invariant-code-motion,
        canonicalize,
        convert-vector-to-scf,
        convert-linalg-to-loops,
        buffer-deallocation-pipeline,
        scf-forall-to-parallel,
        convert-scf-to-openmp,
        expand-strided-metadata,
        finalize-memref-to-llvm,
        convert-scf-to-cf,
        lower-affine,

        convert-openmp-to-llvm,
        convert-vector-to-llvm,
        convert-math-to-llvm,
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
        shared_libs=os.getenv("MLIR_SHARED_LIBS", "").split(","),
    )

    full_function_name = os.path.join(
        cfg.benchmarks_folder_path,
        function_name + ".mlir"
    )
    with open(full_function_name, "r") as f:
        original_code = f.read()

    np_file = np.load(full_function_name + ".npz")
    expected: np.ndarray = np.load(full_function_name + ".npy")

    args_names: list[str] = sorted(
        np_file.files,
        key=lambda s: original_code.index(s)
    )
    args_map: dict[str, np.ndarray] = {arr: np_file[arr] for arr in args_names}
    args = []
    for arg_name in args_names:
        args.append(ctypes.pointer(ctypes.pointer(
            get_ranked_memref_descriptor(args_map[arg_name])
        )))

    delta_arg = (ctypes.c_int64 * 1)(0)
    args.append(delta_arg)

    try:
        execution_engine.invoke("main", *args)
        execution_engine.invoke("main", *args)
    except Exception as e:
        return None, e
    actual = args_map[args_names[-1]]
    if expected.dtype == np.complex128:
        actual = actual.view(np.complex128).squeeze(len(actual.shape) - 1)
    assertion = np.allclose(actual, expected)

    return delta_arg[0], assertion


def evaluate_code_with_bindings_wrapper(code: str, function_name: str, exec_times, assertions):
    """Wrapper function for evaluate_code_with_bindings to be used in multiprocessing.

    Args:
        code (str): The MLIR code to run.
        function_name (str): The name of the function to run.
        exec_times (list): A list to store the execution times.
        assertions (list): A list to store the assertion results
    """
    exec_time, assertion = evaluate_code_with_bindings(code, function_name)
    exec_times.append(exec_time)
    assertions.append(assertion)


def evaluate_code_with_bindings_and_timeout(code: str, function_name: str, timeout: Optional[float]) -> tuple[Optional[int], Union[Exception, bool]]:
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
    process = multiprocessing.Process(target=evaluate_code_with_bindings_wrapper, args=(code, function_name, exec_times, assertions))
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
    command_1 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' -buffer-deallocation -scf-forall-to-parallel -convert-linalg-to-loops  -convert-vector-to-scf -convert-scf-to-openmp -expand-strided-metadata -finalize-memref-to-llvm -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"
    command_2 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs={os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_c_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libomp.so"

    os.environ["OMP_NUM_THREADS"] = "8"

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


def __check_execution_cache(bench_name: str, cache_key: str) -> Optional[int]:
    """Check the execution cache for the given operation state.

    Args:
        cache_key (str): The cache key to check.

    Returns:
        Optional[int]: the execution time in nanoseconds if the operation is found in the cache, otherwise None.
    """
    if not cfg.exec_data_file:
        return None

    # Read json file
    with open(cfg.exec_data_file, "r") as file:
        data = json.load(file)

    if bench_name not in data or cache_key not in data[bench_name]:
        return None

    return int(data[bench_name][cache_key])


def __update_execution_cache(bench_name: str, cache_key: str, exec_time: int):
    """Update the execution cache with the given operation state.

    Args:
        cache_key (str): The cache key to update.
        exec_time (int): The execution time in nanoseconds.
    """
    if not cfg.exec_data_file:
        return

    # Read json file
    with open(cfg.exec_data_file, "r") as file:
        data = json.load(file)

    if bench_name not in data:
        data[bench_name] = {}

    if cache_key in data[bench_name]:
        print_alert("Unexpected hit", data[bench_name][cache_key], exec_time)
        return
    data[bench_name][cache_key] = exec_time

    # Write json file
    with open(cfg.exec_data_file, "w") as file:
        json.dump(data, file, indent=4)


def __get_code_cache_key(state: OperationState, bench_data: BenchmarkFeatures) -> str:
    """Get the code cache key for the given operation state.

    Args:
        state (OperationState): The operation state to get the code cache key.
        bench_data (BenchmarkFeatures): The benchmark features data.

    Returns:
        str: the code cache key.
    """
    trans_codes = {
        'no_transformation': 'N',
        'parallelization': 'P',
        'tiling': 'T',
        'interchange': 'I',
        'vectorization': 'V',
        'img2col': 'C'
    }

    ops_codes = [''] * len(bench_data.operation_tags)
    code = ''
    for transformation, parameters in state.transformation_history:
        if transformation == 'done':
            ops_codes[parameters[0]] = code
            code = ''
            continue

        params_str = ','.join([str(p) for p in parameters])
        code += f'{trans_codes[transformation]}({params_str})'

    assert transformation != 'done'
    last_op_idx = bench_data.operation_tags.index(state.operation_tag)
    ops_codes[last_op_idx] = code

    return '|'.join(ops_codes)
