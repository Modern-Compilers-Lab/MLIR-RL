"""Runner that compiles MLIR to shared library via opt -O3 + llc, with noalias injection.

Combines the -so pipeline (opt -O3 → llc -mcpu=broadwell → .so) with
noalias scope injection for LLVM-level alias analysis.
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
from mlir.runtime import get_ranked_memref_descriptor
from utils import bufferize, lower
from mlir.dialects.func import FuncOp
from inject_noalias import inject_noalias


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

    inputs, outputs, _ = create_params(module)
    expected = np.matmul(inputs[0], inputs[1])
    # Only pass memref args (A, B, C) — exec_time is the return value
    # _mlir_ciface_main takes ptr-to-struct (single pointer), not ptr-to-ptr
    args_list_ctypes = [
        ctypes.pointer(get_ranked_memref_descriptor(arr))
        for arr in inputs + outputs
    ]

    lower(module, args.p)

    # Inject noalias metadata
    asm = module.operation.get_asm()
    modified_asm = inject_noalias(asm)

    # Re-parse with noalias annotations
    with Context() as ctx2:
        ctx2.load_all_available_dialects()
        module = Module.parse(modified_asm)

    # Compile to shared library via opt -O3 + llc
    so_path = compile_to_so(module)

    # Load the shared library
    runner_utils = ctypes.CDLL("/home/mt5383/.conda/envs/main/lib/libmlir_runner_utils.so")
    c_runner_utils = ctypes.CDLL("/home/mt5383/.conda/envs/main/lib/libmlir_c_runner_utils.so")
    omp_lib = ctypes.CDLL("/home/mt5383/.conda/envs/main/lib/libomp.so")
    lib = ctypes.CDLL(so_path)

    # Get the entry function
    func = lib._mlir_ciface_main
    func.restype = ctypes.c_int64

    # Warmup + correctness check
    func(*args_list_ctypes)
    np.testing.assert_allclose(outputs[0], expected)

    for _ in range(10):
        func(*args_list_ctypes)

    times: list[int] = []
    for _ in range(11):
        result = func(*args_list_ctypes)
        times.append(result)
    print(median(times))

    # Cleanup
    os.unlink(so_path)


def compile_to_so(module: Module) -> str:
    """Compile MLIR module to shared library via LLVM IR opt -O3 + llc."""
    tmpdir = tempfile.mkdtemp()
    mlir_file = os.path.join(tmpdir, 'input.mlir')
    ll_file = os.path.join(tmpdir, 'input.ll')
    opt_file = os.path.join(tmpdir, 'opt.ll')
    obj_file = os.path.join(tmpdir, 'output.o')
    so_file = os.path.join(tmpdir, 'output.so')

    # Write MLIR to file
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

    # opt -O3
    base_passes = 'default<O3>'
    extra_passes = os.environ.get('OPT_EXTRA_PASSES', '')
    if extra_passes:
        passes = f'{base_passes},{extra_passes}'
    else:
        passes = base_passes
    opt_cmd = ['opt', f'--passes={passes}']
    llvm_opts = os.environ.get('LLVM_OPTS', '').split()
    opt_cmd.extend(llvm_opts)
    opt_cmd.extend([ll_file, '-S', '-o', opt_file])
    result = subprocess.run(opt_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"opt failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # llc -> object file (target Broadwell)
    llc_cmd = ['llc', '-O3', '-mcpu=broadwell', '-relocation-model=pic']
    llc_opts = os.environ.get('LLC_OPTS', '').split()
    llc_cmd.extend(llc_opts)
    llc_cmd.extend(['-filetype=obj', opt_file, '-o', obj_file])
    result = subprocess.run(llc_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"llc failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Link to shared library
    result = subprocess.run(
        ['gcc', '-shared', '-o', so_file, obj_file,
         '-L/home/mt5383/.conda/envs/main/lib',
         '-lmlir_runner_utils', '-lmlir_c_runner_utils', '-lomp',
         '-Wl,-rpath,/home/mt5383/.conda/envs/main/lib'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"gcc link failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Cleanup intermediate files
    for f in [mlir_file, ll_file, opt_file, obj_file]:
        os.unlink(f)

    return so_file


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


def convert_to_ctypes_args(inputs: list[np.ndarray], outputs: list[np.ndarray], exec_time: np.ndarray) -> list:
    args = [
        ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arr)))
        for arr in inputs + outputs
    ]
    args.append(exec_time.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    return args


if __name__ == "__main__":
    main()
