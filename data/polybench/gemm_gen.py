import numpy as np
from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
from mlir.passmanager import PassManager
import os

bench_name = "gemm"
MATRIX_SIZE = 32
bench_file = f"{bench_name}.mlir.bench"
bench_output = f"{bench_name}_{MATRIX_SIZE}.mlir"

params = {
    "NI": MATRIX_SIZE,
    "NJ": MATRIX_SIZE,
    "NK": MATRIX_SIZE,
}

inputs = {
    'A': np.random.rand(params['NI'], params['NK']) * 100,
    'B': np.random.rand(params['NK'], params['NJ']) * 100,
    'alpha': np.random.rand(1),
    'beta': np.random.rand(1),
    'C': np.random.rand(params['NI'], params['NJ']) * 100,
}

order = ['A', 'B', 'alpha', 'beta', 'C']

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
    code = f.read()
for key, value in params.items():
    code = code.replace(key, str(value))

with open(bench_output, "w") as f:
    f.write(code)
np.savez(f"{bench_output}.npz", **inputs)

C = inputs['C'].copy()
for _ in range(2):
    C = inputs['alpha'] * inputs['A'] @ inputs['B'] + inputs['beta'] * C
expected = C
np.save(f"{bench_output}.npy", expected)

# ------ End of generation ------ #
exit(0)

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
    print(None, e)

actual = inputs[order[-1]]
# if expected.dtype == np.complex128:
#     actual = actual.view(np.complex128).squeeze(len(actual.shape) - 1)
assertion = np.allclose(actual, expected)

print(delta_arg[0], assertion)
