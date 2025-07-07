import os
import numpy as np
from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
from mlir.passmanager import PassManager
from typing import Union, Optional
import multiprocessing
import multiprocessing.managers
from aas import config as cfg
from aas.state import OperationState
from aas.transforms import apply_transformation_with_timeout
from aas.observation.benchmark import BenchmarkFeatures
import json
import re


# ================================== Evaluation Functions (Python Bindings) ==================================

# TODO: Adapt this function to be able to run code without benchmark name
def evaluate_code_with_bindings(code: str, function_name: str) -> tuple[Optional[float], Union[Exception, bool]]:
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

    try:
        execution_engine.invoke("main", *args)
        execution_engine.invoke("main", *args)
    except Exception as e:
        return None, e

    return delta_arg[0], True


def evaluate_code_with_bindings_wrapper(code: str, function_name: str, exec_times: multiprocessing.managers.ListProxy, assertions: multiprocessing.managers.ListProxy):
    """Wrapper function for evaluate_code_with_bindings to be used in multiprocessing.

    Args:
        code (str): The MLIR code to run.
        function_name (str): The name of the function to run.
        exec_times (multiprocessing.managers.ListProxy): A list to store the execution times.
        assertions (multiprocessing.managers.ListProxy): A list to store the assertion results
    """
    exec_time, assertion = evaluate_code_with_bindings(code, function_name)
    exec_times.append(exec_time)
    assertions.append(assertion)


def evaluate_code_with_bindings_and_timeout(code: str, function_name: str, timeout: Optional[float] = None):
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
        process.close()

        return None, False
    else:
        # The function completed within the timeout
        process.close()
        return exec_times[0], assertions[0]


# ================================== Evaluation Functions (MLIR CPU Runner) ==================================

def evaluate_code_with_cmd(code: str, tmp_file_path: str):
    """Lowers and runs the given MLIR code using MLIR opt and MLIR CPU Runner, then returns the execution time and assertion.

    Args:
        code (str): The MLIR code to run.
        tmp_file_path (str): The temporary file path to write the MLIR code.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
    """
    command_1 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt -loop-invariant-code-motion -canonicalize -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' -convert-vector-to-scf -convert-linalg-to-loops -buffer-deallocation-pipeline -scf-forall-to-parallel -convert-scf-to-openmp -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -canonicalize -cse"
    command_2 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs={os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_c_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libomp.so"

    with open(tmp_file_path, "w") as file:
        file.write(code)

    out = os.popen(f"""{command_1} {tmp_file_path} | {command_2} /dev/stdin""").read()

    if out:
        return int(out.strip().split('\n')[-1]), True
    else:
        return None, False


def evaluate_code_with_cmd_wrapper(code: str, tmp_file_path: str, exec_times: multiprocessing.managers.ListProxy, assertions: multiprocessing.managers.ListProxy):
    """Wrapper function for evaluate_code_with_cmd to be used in multiprocessing.

    Args:
        code (str): The MLIR code to run.
        tmp_file_path (str): The temporary file path to write the MLIR code.
        exec_times (multiprocessing.managers.ListProxy): A list to store the execution times.
        assertions (multiprocessing.managers.ListProxy): A list to store the assertion results
    """
    exec_time, assertion = evaluate_code_with_cmd(code, tmp_file_path)
    exec_times.append(exec_time)
    assertions.append(assertion)


def evaluate_code_with_cmd_and_timeout(code: str, tmp_file_path: str, timeout: Optional[float] = None):
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
        process.close()

        return None, False
    else:
        # The function completed within the timeout
        process.close()
        return exec_times[0], assertions[0]


# ================================== Evaluation Functions (Both) ==================================

def evaluate_code_with_timeout(state: OperationState, tmp_file_path: str, timeout: Optional[float] = None, use_cache: bool = True):
    """Evaluates the given MLIR code using Python bindings or MLIR opt and MLIR CPU Runner with a timeout.

    Args:
        state (OperationState): The state to run the Alpha AutoScheduler on.
        tmp_file_path (str): The temporary file path to write the MLIR code.
        timeout (Optional[float]): The timeout in seconds.
        use_cache (bool): Whether to use the execution database to cache execution times.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
        str: the transformed code.
    """
    # Get the code
    code = state.bench_features.code
    # Get the full schedule
    if cfg.optimization_mode == 'all':
        full_schedule = []
        for op_tag in state.bench_features.operation_tags:
            if op_tag == state.operation_tag:
                full_schedule.append(state.transformation_history)
            else:
                full_schedule.append([])
    else:
        full_schedule = [state.transformation_history]
    # Transform the code
    for action in state.transformation_history:
        # If code is not None or empty, apply the transformation
        if code:
            code = apply_transformation_with_timeout(
                state=state,
                code=code,
                tmp_file_path=tmp_file_path,
                action=action,
                timeout=timeout,
                use_vectorizer=cfg.use_vectorizer
            )
        else:
            # If code is None or empty, return execution error
            return None, False, code
    # Check last transformation code
    if not code:
        return None, False, code

    # Check execution database for the execution time of the given state
    if use_cache and cfg.exec_db_path:
        with open(cfg.exec_db_path, "r") as f:
            exec_db = json.load(f)
        bench_db = exec_db.get(state.bench_features.bench_name)
        if bench_db:
            exec_time = bench_db.get(BenchmarkFeatures.any_schedule_to_str(full_schedule))
            if exec_time:
                return exec_time, True, code

    # Otherwise execute the code manually using Python bindings if enabled
    if cfg.use_bindings:
        exec_time, assertion = evaluate_code_with_bindings(code, state.bench_features.bench_name)
    else:
        exec_time, assertion = evaluate_code_with_cmd(code, tmp_file_path)
    # Store the execution time in the execution database
    if use_cache and (exec_time is not None) and assertion and cfg.exec_db_path:
        with open(cfg.exec_db_path, "r") as f:
            exec_db = json.load(f)
        bench_db = exec_db.get(state.bench_features.bench_name)
        if not bench_db:
            bench_db = {}
            exec_db[state.bench_features.bench_name] = bench_db
        exec_db[state.bench_features.bench_name][BenchmarkFeatures.any_schedule_to_str(full_schedule)] = exec_time
        with open(cfg.exec_db_path, "w") as f:
            json.dump(exec_db, f, indent=2)

    # Return the execution time and assertion result and transformed code
    return exec_time, assertion, code


def evaluate_benchmark_code_with_timeout(states: list[OperationState], tmp_file_path: str, timeout: Optional[float] = None, use_cache: bool = True):
    """Evaluates the given MLIR code using Python bindings or MLIR opt and MLIR CPU Runner with a timeout.

    Args:
        states (list[OperationState]): The states to run the Alpha AutoScheduler on.
        tmp_file_path (str): The temporary file path to write the MLIR code.
        timeout (Optional[float]): The timeout in seconds.
        use_cache (bool): Whether to use the execution database to cache execution times.

    Returns:
        Optional[float]: the execution time in seconds.
        bool: the assertion result.
        str: the transformed code.
    """
    if not states:
        return None, False, None
    # Get the code
    bench_features = states[0].bench_features
    code = bench_features.code
    # Get the full schedule
    full_schedule = []
    # Transform the code
    for state in states:
        for action in state.transformation_history:
            # If code is not None or empty, apply the transformation
            if code:
                code = apply_transformation_with_timeout(
                    state=state,
                    code=code,
                    tmp_file_path=tmp_file_path,
                    action=action,
                    timeout=timeout,
                    use_vectorizer=cfg.use_vectorizer
                )
            else:
                # If code is None or empty, return execution error
                return None, False, code
        # Update the full schedule
        full_schedule.insert(0, state.transformation_history)
    # Check last transformation code
    if not code:
        return None, False, code

    # Check execution database for the execution time of the given state
    if use_cache and cfg.exec_db_path:
        with open(cfg.exec_db_path, "r") as f:
            exec_db = json.load(f)
        bench_db = exec_db.get(bench_features.bench_name)
        if bench_db:
            exec_time = bench_db.get(BenchmarkFeatures.any_schedule_to_str(full_schedule))
            if exec_time:
                return exec_time, True, code
    # Otherwise execute the code manually using Python bindings if enabled
    if cfg.use_bindings:
        exec_time, assertion = evaluate_code_with_bindings(code, bench_features.bench_name)
    else:
        exec_time, assertion = evaluate_code_with_cmd(code, tmp_file_path)
    # Store the execution time in the execution database
    if use_cache and (exec_time is not None) and assertion and cfg.exec_db_path:
        with open(cfg.exec_db_path, "r") as f:
            exec_db = json.load(f)
        bench_db = exec_db.get(bench_features.bench_name)
        if not bench_db:
            bench_db = {}
            exec_db[bench_features.bench_name] = bench_db
        exec_db[bench_features.bench_name][BenchmarkFeatures.any_schedule_to_str(full_schedule)] = exec_time
        with open(cfg.exec_db_path, "w") as f:
            json.dump(exec_db, f, indent=2)

    # Return the execution time and assertion result and transformed code
    return exec_time, assertion, code


def get_cached_exec_time(exec_db: Optional[dict], state: OperationState):
    """Get the cached execution time of the given state.

    Args:
        exec_db (Optional[dict]): The execution database for the benchmark.
        state (OperationState): The state to get the execution time of.

    Returns:
        Optional[float]: the cached execution time in seconds.
    """
    # Get the full schedule
    if cfg.optimization_mode == 'all':
        full_schedule = []
        for op_tag in state.bench_features.operation_tags:
            if op_tag == state.operation_tag:
                full_schedule.append(state.transformation_history)
            else:
                full_schedule.append([])
    else:
        full_schedule = [state.transformation_history]
    # Check execution database for the execution time of the given state
    if exec_db is not None:
        return exec_db.get(BenchmarkFeatures.any_schedule_to_str(full_schedule))
    # Else return None
    return None


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
