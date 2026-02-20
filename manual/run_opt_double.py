"""Run MLIR matmul with double opt -O3 pre-processing.

Pipeline: MLIR → lower → mlir-translate → opt -O3 → opt -O3 → mlir-translate -import-llvm → JIT(opt_level=3)
"""

import argparse
import ctypes
import ctypes.util
import os
import subprocess
import tempfile
from statistics import median
import numpy as np

from mlir._mlir_libs._mlir.ir import Context, Module, MemRefType, IntegerType, F64Type, F32Type
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor
from utils import bufferize, lower
from mlir.dialects.func import FuncOp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True, help='Pass pipeline file')
    parser.add_argument('code_file', nargs='?', help='MLIR code file (reads stdin if omitted)')
    args = parser.parse_args()

    if args.code_file:
        with open(args.code_file, 'r') as f:
            code = f.read()
    else:
        import sys
        code = sys.stdin.read()

    with Context() as ctx:
        ctx.load_all_available_dialects()
        module = Module.parse(code)

    bufferize(module)

    inputs, outputs, exec_time = create_params(module)
    expected = np.matmul(inputs[0], inputs[1])
    args_list = convert_to_args(inputs, outputs, exec_time)

    lower(module, args.p)

    asm = module.operation.get_asm()

    with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as f:
        f.write(asm)
        mlir_file = f.name

    with tempfile.NamedTemporaryFile(suffix='.ll', mode='w', delete=False) as f:
        ll_file = f.name

    with tempfile.NamedTemporaryFile(suffix='.ll', mode='w', delete=False) as f:
        opt_file = f.name

    with tempfile.NamedTemporaryFile(suffix='.ll', mode='w', delete=False) as f:
        opt_file2 = f.name

    with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as f:
        reimport_file = f.name

    try:
        result = subprocess.run(
            ['mlir-translate', '-mlir-to-llvmir', mlir_file, '-o', ll_file],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'mlir-translate failed: {result.stderr}')

        # First opt -O3
        result = subprocess.run(
            ['opt', '--passes=default<O3>', ll_file, '-S', '-o', opt_file],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'opt pass 1 failed: {result.stderr}')

        # Second opt -O3
        result = subprocess.run(
            ['opt', '--passes=default<O3>', opt_file, '-S', '-o', opt_file2],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'opt pass 2 failed: {result.stderr}')

        result = subprocess.run(
            ['mlir-translate', '-import-llvm', opt_file2, '-o', reimport_file],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'mlir-translate import failed: {result.stderr}')

        with open(reimport_file) as f:
            opt_mlir = f.read()

        with Context() as ctx2:
            ctx2.load_all_available_dialects()
            module = Module.parse(opt_mlir)

    finally:
        for f in [mlir_file, ll_file, opt_file, opt_file2, reimport_file]:
            try:
                os.unlink(f)
            except OSError:
                pass

    import sys
    print('double opt -O3 pipeline complete', file=sys.stderr)

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
