"""Run script for tensor-land pad+hoist schedules.

Handles memref.copy → linalg.copy conversion needed when tensor.pad nofold
creates memref.copy ops with strided memrefs. These strided copies can't be
lowered by finalize-memref-to-llvm (creates unrealized_conversion_cast).

Approach: after bufferization, text-replace memref.copy with linalg.copy,
then convert-linalg-to-loops lowers them to explicit load/store loops.
"""
import argparse
import ctypes
import ctypes.util
import os
import re
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
from mlir.passmanager import PassManager
from mlir.dialects.func import FuncOp


def replace_memref_copy_with_linalg_copy(ir_text: str) -> str:
    """Replace memref.copy with linalg.copy in MLIR IR text.

    memref.copy %src, %dst : <src_type> to <dst_type>
    → linalg.copy ins(%src : <src_type>) outs(%dst : <dst_type>)

    Self-copies (%x, %x) are replaced with nothing (they're no-ops).
    """
    lines = ir_text.split('\n')
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('memref.copy '):
            # Parse: memref.copy %src, %dst : src_type to dst_type
            rest = stripped[len('memref.copy '):]
            # Find the two operands
            comma_idx = rest.index(',')
            src = rest[:comma_idx].strip()
            after_comma = rest[comma_idx + 1:]
            # Find the ' : '
            colon_idx = after_comma.index(' : ')
            dst = after_comma[:colon_idx].strip()
            type_part = after_comma[colon_idx + 3:]
            # Find ' to ' separator between types
            to_idx = type_part.index(' to ')
            src_type = type_part[:to_idx].strip()
            dst_type = type_part[to_idx + 4:].strip()

            if src == dst:
                # Self-copy is a no-op, skip it
                continue

            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f'{indent}linalg.copy ins({src} : {src_type}) outs({dst} : {dst_type})')
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)


def bufferize_with_linalg_copy(module: Module):
    """Bufferize, then replace memref.copy with linalg.copy via text rewrite."""
    pass_pipeline = """builtin.module(
        eliminate-empty-tensors,
        empty-tensor-to-alloc-tensor,
        one-shot-bufferize{
            bufferize-function-boundaries
            unknown-type-conversion=identity-layout-map
            function-boundary-type-conversion=identity-layout-map
        },
        buffer-results-to-out-params{hoist-static-allocs add-result-attr},
        canonicalize, cse
    )"""
    pm = PassManager.parse(pass_pipeline, module.context)
    pm.run(module.operation)

    ir_text = str(module)
    new_ir = replace_memref_copy_with_linalg_copy(ir_text)

    new_module = Module.parse(new_ir, module.context)
    return new_module


def lower(module: Module, pass_file: str):
    with open(pass_file) as f:
        pipeline = f.read()
    pm = PassManager.parse(pipeline, module.context)
    pm.run(module.operation)


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

        module = bufferize_with_linalg_copy(module)

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
    args: list = [
        ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arr)))
        for arr in inputs + outputs
    ]
    args.append(exec_time.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    return args


if __name__ == "__main__":
    main()
