import numpy as np
from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
from mlir.passmanager import PassManager
import os

def print_info(*args):
    message = ' '.join(map(str, args))
    print(f"\033[94m[INFO]\t {message}\033[0m")

def print_success(*args):
    message = ' '.join(map(str, args))
    print(f"\033[92m[SUCCESS]\t {message}\033[0m")

def print_alert(*args):
    message = ' '.join(map(str, args))
    print(f"\033[93m[ALERT]\t {message}\033[0m")

def print_error(*args):
    message = ' '.join(map(str, args))
    print(f"\033[91m[ERROR]\t {message}\033[0m")

def lower_and_run_code(code: str, function_name: str) -> float:
    pass_pipeline = """builtin.module(
        loop-invariant-code-motion,
        canonicalize,
        convert-vector-to-scf,
        convert-linalg-to-loops,
        buffer-deallocation-pipeline,
        convert-scf-to-openmp,
        expand-strided-metadata,
        finalize-memref-to-llvm,
        convert-scf-to-cf,
        lower-affine,

        convert-math-to-llvm,
        convert-vector-to-llvm,
        convert-func-to-llvm,
        convert-index-to-llvm,
        convert-arith-to-llvm,
        convert-openmp-to-llvm,
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
        "lqcd-benchmarks",
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

    execution_engine.invoke("main", *args)
    execution_engine.invoke("main", *args)
    actual = args_map[args_names[-1]]
    if expected.dtype == np.complex128:
        actual = actual.view(np.complex128).squeeze(len(actual.shape) - 1)
    assertion = np.allclose(actual, expected)
    if not assertion:
        print_error(f"ASSERTION FAILED: {actual} != {expected}")

    return delta_arg[0] / 1e9
