import numpy as np
from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
from mlir.passmanager import PassManager
import os

bench_name = "fdtd"
MATRIX_SIZE = 2048
TMAX = 50
assert TMAX != 0 and TMAX != 1
bench_file = f"{bench_name}.mlir.bench"
bench_output = f"{bench_name}_{MATRIX_SIZE}_{TMAX}.mlir"

params = {
    "NX0": MATRIX_SIZE,
    "NY0": MATRIX_SIZE,
    "NX1": MATRIX_SIZE - 1,
    "NY1": MATRIX_SIZE - 1,
    "TMAX": TMAX,
}

inputs = {
    'EX': np.random.rand(MATRIX_SIZE, MATRIX_SIZE) * 100,
    'EY': np.random.rand(MATRIX_SIZE, MATRIX_SIZE) * 100,
    'HZ': np.random.rand(MATRIX_SIZE, MATRIX_SIZE) * 100,
}

order = ['EX', 'EY', 'HZ']

with open(bench_file, "r") as f:
    code = f.read()
for key, value in params.items():
    code = code.replace(key, str(value))

with open(bench_output, "w") as f:
    f.write(code)
np.savez(f"{bench_output}.npz", **inputs)

EY = inputs['EY'].copy()
EX = inputs['EX'].copy()
HZ = inputs['HZ'].copy()
for _ in range(2):
    for t in range(TMAX):
        EY[0, :] = t
        EY[1:, :] = EY[1:, :] - 0.5 * (HZ[1:, :] - HZ[:-1, :])
        EX[:, 1:] = EX[:, 1:] - 0.5 * (HZ[:, 1:] - HZ[:, :-1])
        HZ[:-1, :-1] = HZ[:-1, :-1] - 0.7 * (
            EX[:-1, 1:] - EX[:-1, :-1] + EY[1:, :-1] - EY[:-1, :-1]
        )
expected = HZ
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

actual = inputs['HZ']
# if expected.dtype == np.complex128:
#     actual = actual.view(np.complex128).squeeze(len(actual.shape) - 1)
assertion = np.allclose(actual, expected)
print(delta_arg[0], assertion)
