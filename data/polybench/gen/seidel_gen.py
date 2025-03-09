import numpy as np
from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
from mlir.passmanager import PassManager
import os
import sys
import json

base_name = "seidel"
bench_file = f"{base_name}.mlir.bench"
order = ['B', 'A']

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

with open(bench_file, "r") as f:
    base_code = f.read()

execution_times = {}

for TSTEPS in [2, 10, 20, 50, 100]:
    for i in range(5, 13):
        MATRIX_SIZE = 2 ** i
        bench_name = f"{base_name}_{MATRIX_SIZE}_{TSTEPS}"
        bench_output = f"../{bench_name}.mlir"

        params = {
            "N0": MATRIX_SIZE,
            "N_2": MATRIX_SIZE - 2,
            "N1": MATRIX_SIZE + 1,
            'N2': MATRIX_SIZE + 2,
            'N3': MATRIX_SIZE * 2,
            'N4': MATRIX_SIZE * 2 + 1,
            'N5': MATRIX_SIZE * 2 + 2,
            "TSTEPS": TSTEPS,
        }

        inputs = {
            'A': np.random.rand(MATRIX_SIZE, MATRIX_SIZE) * 100,
            'B': np.zeros((MATRIX_SIZE, MATRIX_SIZE)),
        }
        A = inputs['A'].copy()
        for _ in range(2):
            for _ in range(TSTEPS):
                A[1:-1, 1:-1] = (A[:-2, :-2] + A[:-2, 1:-1] + A[:-2, 2:] + A[1:-1, :-2] + A[1:-1, 1:-1] + A[1:-1, 2:] + A[2:, :-2] + A[2:, 1:-1] + A[2:, 2:]) / 9.0
        expected = A
        np.savez(f"{bench_output}.npz", **inputs)

        code = base_code
        for key, value in params.items():
            code = code.replace(key, str(value))

        with Context():
            module = Module.parse(code)
            pm = PassManager.parse(pass_pipeline)
            pm.run(module.operation)
        execution_engine = ExecutionEngine(
            module,
            shared_libs=os.getenv("MLIR_SHARED_LIBS", "").split(","),
        )

        args = []
        for arg_name in order:
            args.append(ctypes.pointer(ctypes.pointer(
                get_ranked_memref_descriptor(inputs[arg_name])
            )))

        delta_arg = (ctypes.c_int64 * 1)(0)
        args.append(delta_arg)

        try:
            execution_engine.invoke("main", *args)
            execution_engine.invoke("main", *args)
        except Exception as e:
            print("Benchmark failed:", bench_name, e, file=sys.stderr)
            os.remove(f"{bench_output}.npz")
            continue

        exec_time = delta_arg[0]
        if exec_time >= (1 * 10**9):
            os.remove(f"{bench_output}.npz")
            break

        actual = inputs[order[-1]]
        assertion = np.allclose(actual, expected)
        if not assertion:
            print("Assertion failed:", bench_name, file=sys.stderr)
            os.remove(f"{bench_output}.npz")
            continue

        with open(bench_output, "w") as f:
            f.write(code)
        np.save(f"{bench_output}.npy", expected)
        execution_times[bench_name] = exec_time

with open('../execution_times.json', 'r') as f:
    data: dict = json.load(f)
data.update(execution_times)
with open('../execution_times.json', 'w') as f:
    json.dump(data, f, indent=4)
