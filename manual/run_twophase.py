"""Run MLIR matmul with two-phase transform: tensor-land then memref-land.

Phase 1: Transform schedule applied on tensor IR (pad + hoist_pad) — done externally by mlir-opt
Phase 2: The input is already bufferized memref IR with linalg.matmul remaining.
         Apply a second transform (micro-tile + vectorize) then lower and JIT.

Usage: pipe already-bufferized MLIR (with linalg.matmul remaining) to stdin.
  mlir-opt ... | python run_twophase.py -p <lowering_pipeline> -s <second_schedule>
"""

import argparse
import ctypes
import ctypes.util
import os
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
from mlir.passmanager import PassManager
from mlir.runtime import get_ranked_memref_descriptor
from utils import lower
from mlir.dialects.func import FuncOp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True, help='Pass pipeline file')
    parser.add_argument('-s', required=False, help='Second-phase schedule file (mlir transform)')
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

    # Apply second-phase transform if provided
    if args.s:
        transform_pipeline = (
            f'transform-preload-library{{transform-library-paths={args.s}}},'
            f'transform-interpreter'
        )
        pm = PassManager.parse(f'builtin.module({transform_pipeline})', module.context)
        pm.run(module.operation)

    inputs, outputs, exec_time = create_params(module)
    expected = np.matmul(inputs[0], inputs[1])
    args_list = convert_to_args(inputs, outputs, exec_time)

    lower(module, args.p)

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
