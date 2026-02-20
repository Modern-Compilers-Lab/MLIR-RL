"""Runner that applies LLVM LICM pass to promote C accumulators to registers.

Pipeline: MLIR -> lower -> mlir-translate -> opt (LICM only) -> mlir-translate -import-llvm -> JIT
"""
import argparse
import ctypes
import ctypes.util
import os
import subprocess
import sys
import tempfile
from statistics import median

import numpy as np

from mlir._mlir_libs._mlir.ir import Context, Module, MemRefType, IntegerType, F64Type, F32Type  # type: ignore
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
        code = sys.stdin.read()

    with Context() as ctx:
        ctx.load_all_available_dialects()
        module = Module.parse(code)

    bufferize(module)

    inputs, outputs, exec_time = create_params(module)
    expected = np.matmul(inputs[0], inputs[1])
    args_list = convert_to_args(inputs, outputs, exec_time)

    lower(module, args.p)

    # Apply LICM via opt to promote C accumulators to registers
    module = apply_licm(module)

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


def apply_licm(module: Module) -> Module:
    """Apply LLVM LICM pass via opt to promote C stores/loads out of K-loop."""
    tmpdir = tempfile.mkdtemp()
    mlir_file = os.path.join(tmpdir, 'input.mlir')
    ll_file = os.path.join(tmpdir, 'input.ll')
    licm_file = os.path.join(tmpdir, 'licm.ll')
    reimport_file = os.path.join(tmpdir, 'reimport.mlir')

    with open(mlir_file, 'w') as f:
        f.write(str(module.operation))

    # MLIR -> LLVM IR
    result = subprocess.run(
        ['mlir-translate', '-mlir-to-llvmir', mlir_file, '-o', ll_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"mlir-translate failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Apply LICM only (not full O3)
    result = subprocess.run(
        ['opt', '--passes=loop-mssa(licm)', ll_file, '-S', '-o', licm_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"opt licm failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Re-import to MLIR
    result = subprocess.run(
        ['mlir-translate', '-import-llvm', licm_file, '-o', reimport_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"mlir-translate import failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse re-imported module
    with open(reimport_file) as f:
        reimported = f.read()

    with Context() as ctx:
        ctx.load_all_available_dialects()
        new_module = Module.parse(reimported)

    # Cleanup
    for f_path in [mlir_file, ll_file, licm_file, reimport_file]:
        os.unlink(f_path)
    os.rmdir(tmpdir)

    return new_module


def create_params(module: Module):
    def get_dtype(memref_type: MemRefType):
        et = memref_type.element_type
        match et:
            case F64Type():
                np_dtype = np.float64
            case F32Type():
                np_dtype = np.float32
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
        raise Exception(f'unexpected result types {res_types}')

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
