import numpy as np
from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
from mlir.passmanager import PassManager
import os

bench_name = "floyd"
MATRIX_SIZE = 2048
bench_file = f"{bench_name}.mlir.bench"
bench_output = f"{bench_name}_{MATRIX_SIZE}.mlir"

params = {
    "N": MATRIX_SIZE,
}

inputs = {
    'P': np.random.rand(MATRIX_SIZE, MATRIX_SIZE) * 100,
}

order = ['P']

with open(bench_file, "r") as f:
    code = f.read()
for key, value in params.items():
    code = code.replace(key, str(value))

with open(bench_output, "w") as f:
    f.write(code)
np.savez(f"{bench_output}.npz", **inputs)

P = inputs['P'].copy()
for k in range(MATRIX_SIZE):
    P[:, :] = np.minimum(P[:, :], P[:, k].reshape(-1, 1) + P[k, :].reshape(1, -1))
expected = P
np.save(f"{bench_output}.npy", expected)

# ------ End of generation ------ #
exit(0)

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

actual = inputs['P']
# if expected.dtype == np.complex128:
#     actual = actual.view(np.complex128).squeeze(len(actual.shape) - 1)
assertion = np.allclose(actual, expected)
print(delta_arg[0], assertion)
