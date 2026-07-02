import os
import re
import traceback
from filelock import FileLock
import numpy as np
import json
import multiprocessing

from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
from mlir.passmanager import PassManager

from typing import Optional

from rl_autoschedular_v4_9 import config as cfg
from utils.log import print_alert, print_error,stable_hash,open_cache_db

def get_cached_execution_time(transformed_code: str) -> Optional[int]:
    if not cfg.cache_file:
        return None
    
    lock = FileLock(f"{cfg.cache_file}.lock")
    with lock:
        with open(cfg.cache_file,"r") as f:
            try:
                exec_cache: dict[str, int] = json.load(f)
            except json.JSONDecodeError as e:
                print_error(f"Json read failure: {e}")
                exec_cache = {}
    
    key = str(stable_hash(transformed_code)) # JSON only accepts strings as keys
    
    return exec_cache.get(key) # None if key is not set

def set_cached_execution_time(transformed_code: str, execution_time: int):
    if not cfg.cache_file:
        return
    
    lock = FileLock(f"{cfg.cache_file}.lock")
    
    with lock:
        with open(cfg.cache_file,"r+") as f:
            try:
                exec_cache: dict[str, int] = json.load(f)
            except json.JSONDecodeError as e:
                print_error(f"Json read failure: {e}")
                exec_cache = {}
        
            key = stable_hash(transformed_code) # JSON automatically transforms keys to string type
            exec_cache[key] = execution_time

            f.seek(0)
            f.truncate()

            json.dump(exec_cache, f)


def set_cached_execution_time_sqlite(transformed_code: str, execution_time: int):
    if not cfg.cache_file:
        return

    code_hash = stable_hash(transformed_code)

    with open_cache_db(cfg.cache_file) as conn:
        conn.execute(
            """INSERT INTO execution_cache (code_hash, execution_time) VALUES (?, ?)""", 
            (code_hash, execution_time)
        )

def get_cached_execution_time_sqlite(transformed_code: str) -> Optional[int]:
    if not cfg.cache_file:
        return None

    code_hash = stable_hash(transformed_code)

    with open_cache_db(cfg.cache_file) as conn:
        cursor = conn.execute(
            """
                SELECT execution_time
                FROM execution_cache
                WHERE code_hash = ?
            """, (code_hash,)
        )
        
        row = cursor.fetchone()
        return row[0] if row else None


# ================================== Evaluation Functions (Python Bindings) ==================================

def evaluate_code_with_bindings(code: str) -> tuple[Optional[int], bool]:
    """Lowers and runs the given MLIR code using Python bindings, then returns the execution time and assertion
    result (if the executed code returns the correct result).

    Args:
        code (str): The MLIR code to run.

    Returns:
        Optional[float]: the execution time in nanoseconds (depends on what is returned by the code).
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
        convert-vector-to-scf,
        convert-linalg-to-loops,
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
        convert-math-to-libm,
        finalize-memref-to-llvm,
        convert-func-to-llvm,
        convert-index-to-llvm,
        convert-arith-to-llvm,
        convert-cf-to-llvm,

        reconcile-unrealized-casts,
        canonicalize,
        cse
    )"""

    os.environ["OMP_NUM_THREADS"] = str(cfg.openmp_num_threads)

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
        [execution_engine.invoke("main", *args) for _ in range(1)]
    except Exception as e:
        print(e)
        traceback.print_exc()    
        return None, False

    if delta_arg[0] is None:
        print("",end="")

    return delta_arg[0], True

def evaluate_code_with_bindings_wrapper(code: str, exec_times, assertions):
    """Wrapper function for evaluate_code_with_bindings to be used in multiprocessing.

    Args:
        code (str): The MLIR code to run.
        function_name (str): The name of the function to run.
        exec_times (list): A list to store the execution times.
        assertions (list): A list to store the assertion results
    """
    try:
        exec_time, assertion = evaluate_code_with_bindings(code)
        exec_times.append(exec_time)
        assertions.append(assertion)
    except Exception as e:
        print(e)
        traceback.print_exc()        

def evaluate_code_with_bindings_and_timeout(code: str, timeout: Optional[float]) -> tuple[Optional[int], bool]:
    """Evaluates the given MLIR code using Python bindings with a timeout.

    Args:
        code (str): The MLIR code to run.
        function_name (str): The name of the function to run.
        timeout (Optional[float]): The timeout in seconds.

    Returns:
        Optional[float]: the execution time in nanoseconds.
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
        print_alert("timeout")
        return None, False
    else:
        # The function completed within the timeout
        if not (exec_times and assertions):
            print("", end="")
        
        return exec_times[0] if exec_times else 0, assertions[0] if assertions else False

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
    command_1 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt  -loop-invariant-code-motion -canonicalize -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' -convert-vector-to-scf -convert-linalg-to-loops -buffer-deallocation-pipeline -scf-forall-to-parallel -convert-scf-to-openmp -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm  -convert-math-to-libm -convert-func-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts -canonicalize -cse"
    command_2 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs={os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_c_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libomp.so"

    os.environ["OMP_NUM_THREADS"] = str(cfg.openmp_num_threads)

    lock = FileLock(f"{tmp_file_path}.lock")
    with lock:
        with open(tmp_file_path, "w") as file:
            file.write(code)

        out = os.popen(f"""{command_1} {tmp_file_path} | {command_2} /dev/stdin""").read()

    if out:
        return int(out.strip().split('\n')[-1]), True
    else:
        return None, False

def evaluate_transform_with_cmd(code: str, tmp_file_path: str):
    """Only lowers the given MLIR code using MLIR opt, then returns the new code.

    Args:
        code (str): The MLIR code to run.
        tmp_file_path (str): The temporary file path to write the MLIR code.

    Returns:
        Optional[str]: the new code.
    """
    command_1 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt  -loop-invariant-code-motion -canonicalize -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' -convert-vector-to-scf -convert-linalg-to-loops"
    # command_2 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs={os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_c_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libomp.so"

    os.environ["OMP_NUM_THREADS"] = str(cfg.openmp_num_threads)

    with open(tmp_file_path, "w") as file:
        file.write(code)

    out = os.popen(f"""{command_1} {tmp_file_path}""").read()

    if out:
        return out 
    else:
        return None # maybe make it return code


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

        return None, False
    else:
        # The function completed within the timeout
        return exec_times[0], assertions[0]


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
