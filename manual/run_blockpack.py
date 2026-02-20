"""Run MLIR matmul with block-packed layout.

Pipeline:
  1. mlir-opt -linalg-block-pack-matmul (tensor-level packing)
  2. mlir-opt -transform-interpreter (tensor-level tiling+vectorization)
  3. bufferize
  4. mlir-opt -pass-pipeline (lowering)
  5. JIT execution
"""

import argparse
import ctypes
import ctypes.util
import os
import subprocess
import tempfile
from statistics import median
import numpy as np

# Set LLVM CL options BEFORE any MLIR imports/initialization
if os.environ.get('LLVM_OPTS'):
    try:
        import mlir._mlir_libs
        _lib_dir = os.path.dirname(mlir._mlir_libs.__file__)
        _lib = ctypes.CDLL(os.path.join(_lib_dir, 'libMLIRPythonCAPI.so'))
        _func = _lib.LLVMParseCommandLineOptions
        _func.restype = None
        _func.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_char_p]
        _opts = [b'mlir'] + [o.encode() for o in os.environ['LLVM_OPTS'].split()]
        _argc = len(_opts)
        _argv_arr = (ctypes.c_char_p * _argc)(*_opts)
        _func(_argc, _argv_arr, None)
    except Exception as e:
        import sys
        print(f"Warning: Failed to set LLVM options: {e}", file=sys.stderr)

from mlir._mlir_libs._mlir.ir import Context, Module, MemRefType, IntegerType, F64Type, F32Type  # type: ignore
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor
from mlir.dialects.func import FuncOp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True, help='Pass pipeline file (for lowering)')
    parser.add_argument('-s', required=True, help='Schedule MLIR file (transform)')
    parser.add_argument('--block-factors', default='32,8,256', help='Block factors mb,nb,kb')
    parser.add_argument('code_file', nargs='?', help='MLIR code file (reads stdin if omitted)')
    args = parser.parse_args()

    if args.code_file:
        with open(args.code_file, 'r') as f:
            code = f.read()
    else:
        import sys
        code = sys.stdin.read()

    # Step 1: Block-pack the matmul (tensor-level)
    with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as f:
        f.write(code)
        input_file = f.name

    with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as f:
        packed_file = f.name

    try:
        result = subprocess.run(
            ['mlir-opt', f'-linalg-block-pack-matmul=block-factors={args.block_factors}',
             input_file, '-o', packed_file],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'block-pack-matmul failed: {result.stderr}')

        # Step 2: Apply transform schedule (tensor-level)
        with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as f:
            transformed_file = f.name

        result = subprocess.run(
            ['mlir-opt',
             f'-transform-preload-library=transform-library-paths={args.s}',
             '-transform-interpreter',
             packed_file, '-o', transformed_file],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'transform-interpreter failed: {result.stderr}')

        # Step 3: Bufferize
        with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as f:
            bufferized_file = f.name

        result = subprocess.run(
            ['mlir-opt',
             '-eliminate-empty-tensors', '-empty-tensor-to-alloc-tensor',
             '-one-shot-bufferize=unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries',
             '-buffer-results-to-out-params=hoist-static-allocs add-result-attr',
             '-promote-buffers-to-stack=max-alloc-size-in-bytes=262144 max-rank-of-allocated-memref=4',
             '-canonicalize', '-cse',
             transformed_file, '-o', bufferized_file],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'bufferize failed: {result.stderr}')

        # Step 4: Read bufferized MLIR
        with open(bufferized_file) as f:
            buf_code = f.read()

    finally:
        for fn in [input_file, packed_file, transformed_file, bufferized_file]:
            try:
                os.unlink(fn)
            except OSError:
                pass

    # Parse, lower, and run - all within the same context
    with Context() as ctx:
        ctx.load_all_available_dialects()
        module = Module.parse(buf_code)

        inputs, outputs, exec_time = create_params(module)
        expected = np.matmul(inputs[0], inputs[1])
        args_list = convert_to_args(inputs, outputs, exec_time)

        # Step 5: Lower using pipeline from .txt file
        with open(args.p) as f:
            pipeline = f.read().strip()

        import mlir.passmanager as pm
        pm_obj = pm.PassManager.parse(pipeline)
        pm_obj.run(module.operation)

        import sys
        print('block-pack pipeline complete', file=sys.stderr)

        execution_engine = ExecutionEngine(
            module,
            opt_level=3,
            shared_libs=[
                "/home/mt5383/.conda/envs/main/lib/libmlir_runner_utils.so",
                "/home/mt5383/.conda/envs/main/lib/libmlir_c_runner_utils.so",
                "/home/mt5383/.conda/envs/main/lib/libomp.so"
            ],
        )

        execution_engine.invoke("main", *args_list)
        np.testing.assert_allclose(outputs[0], expected)

        for _ in range(10):
            execution_engine.invoke("main", *args_list)

        times: list[int] = []
        for _ in range(11):
            execution_engine.invoke("main", *args_list)
            times.append(exec_time.item())
        print(median(times))


def create_params(module: Module):
    def get_dtype(memref_type: MemRefType):
        et = memref_type.element_type
        match et:
            case F32Type():
                np_dtype = np.float32
            case F64Type():
                np_dtype = np.float64
            case IntegerType():
                match et.width:
                    case 32:
                        np_dtype = np.int32
                    case 64:
                        np_dtype = np.int64
                    case _:
                        raise Exception(f'unexpected element type {et}')
            case _:
                raise Exception(f'unexpected element type {et}')
        return np_dtype

    main_func = next(op for op in module.body.operations if isinstance(op, FuncOp) and (op.name.value == 'main'))

    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    for input_type, input_attrs in zip(main_func.type.inputs, main_func.arg_attrs):
        assert isinstance(input_type, MemRefType), f'unexpected input type {input_type}'
        if "bufferize.result" in input_attrs:
            out_arr = np.empty(input_type.shape, dtype=get_dtype(input_type))
            outputs.append(out_arr)
        else:
            in_arr = np.full(input_type.shape, 2, dtype=get_dtype(input_type))
            inputs.append(in_arr)

    res_types = main_func.type.results

    if not (len(res_types) == 1 and isinstance(res_types[0], IntegerType) and res_types[0].width == 64):
        raise Exception(f'unexpected result types {res_types}, expected a single i64 result for execution time')

    exec_time = np.zeros((), dtype=np.int64)

    return inputs, outputs, exec_time


def convert_to_args(inputs: list[np.ndarray], outputs: list[np.ndarray], exec_time: np.ndarray) -> list:
    args: list[ctypes._Pointer[ctypes._Pointer[ctypes.Structure]]] = [
        ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arr)))
        for arr in inputs + outputs
    ]
    args.append(exec_time.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    return args


if __name__ == "__main__":
    main()
