"""Run MLIR matmul with noalias + external opt + LLC direct compilation.

Pipeline: MLIR → lower → inject noalias → mlir-translate → opt → llc → .o → .so → dlopen
"""

import argparse
import ctypes
import ctypes.util
import os
import subprocess
import tempfile
from statistics import median
import numpy as np
import time

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
        import sys
        code = sys.stdin.read()

    with Context() as ctx:
        ctx.load_all_available_dialects()
        module = Module.parse(code)

    bufferize(module)

    inputs, outputs, exec_time = create_params(module)
    expected = np.matmul(inputs[0], inputs[1])

    lower(module, args.p)

    # Inject noalias metadata
    asm = module.operation.get_asm()
    asm = inject_noalias(asm)

    # Pipeline: noalias MLIR → LLVM IR → opt → llc → .o → .so
    tmpdir = tempfile.mkdtemp()
    mlir_file = os.path.join(tmpdir, 'input.mlir')
    ll_file = os.path.join(tmpdir, 'input.ll')
    opt_file = os.path.join(tmpdir, 'opt.ll')
    obj_file = os.path.join(tmpdir, 'output.o')
    so_file = os.path.join(tmpdir, 'output.so')

    try:
        with open(mlir_file, 'w') as f:
            f.write(asm)

        # Step 1: MLIR to LLVM IR
        result = subprocess.run(
            ['mlir-translate', '-mlir-to-llvmir', mlir_file, '-o', ll_file],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'mlir-translate failed: {result.stderr}')

        # Step 2: opt (default O3, configurable via OPT_PASSES)
        passes = os.environ.get('OPT_PASSES', 'default<O3>')
        opt_cmd = ['opt', f'--passes={passes}']
        llvm_opts = os.environ.get('LLVM_OPTS', '').split()
        opt_cmd.extend(llvm_opts)
        opt_cmd.extend([ll_file, '-S', '-o', opt_file])
        result = subprocess.run(opt_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'opt failed: {result.stderr}')

        # Step 3: LLC to object file
        llc_flags = os.environ.get('LLC_FLAGS', '-O3 -relocation-model=pic').split()
        llc_cmd = ['llc'] + llc_flags
        # Also pass LLVM_OPTS to LLC for codegen flags
        llc_cmd.extend(llvm_opts)
        llc_cmd.extend([opt_file, '-filetype=obj', '-o', obj_file])
        result = subprocess.run(llc_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'llc failed: {result.stderr}')

        # Step 4: Link to shared library
        runner_utils = "/home/mt5383/.conda/envs/main/lib/libmlir_runner_utils.so"
        c_runner_utils = "/home/mt5383/.conda/envs/main/lib/libmlir_c_runner_utils.so"
        omp_lib = "/home/mt5383/.conda/envs/main/lib/libomp.so"
        link_cmd = [
            'gcc', '-shared', '-o', so_file, obj_file,
            f'-L{os.path.dirname(runner_utils)}',
            '-lmlir_runner_utils', '-lmlir_c_runner_utils', '-lomp',
            '-lm', '-Wl,-rpath,' + os.path.dirname(runner_utils)
        ]
        result = subprocess.run(link_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'link failed: {result.stderr}')

        # Step 5: Load shared library and get function pointer
        # Load dependencies first
        ctypes.CDLL(runner_utils, mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(c_runner_utils, mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(omp_lib, mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(so_file)

        # The MLIR-generated function is _mlir_ciface_main
        func = lib._mlir_ciface_main
        # Signature: i64 _mlir_ciface_main(memref_desc*, memref_desc*, memref_desc*)
        # After buffer-results-to-out-params: main(A, B, C) -> i64

        import sys
        print('noalias + direct compilation pipeline complete', file=sys.stderr)

        # Build args: same as MLIR execution engine expects
        # _mlir_ciface_main takes pointers to ranked memref descriptors + pointer to i64 result
        arg_ptrs = []
        for arr in inputs + outputs:
            desc = get_ranked_memref_descriptor(arr)
            p = ctypes.pointer(desc)
            arg_ptrs.append(p)

        # Call function: returns i64 execution time
        func.restype = ctypes.c_int64
        func.argtypes = [ctypes.c_void_p] * len(arg_ptrs)

        # Warmup + correctness check
        result_time = func(*[ctypes.cast(p, ctypes.c_void_p) for p in arg_ptrs])
        np.testing.assert_allclose(outputs[0], expected)

        for _ in range(10):
            func(*[ctypes.cast(p, ctypes.c_void_p) for p in arg_ptrs])

        times = []
        for _ in range(11):
            t = func(*[ctypes.cast(p, ctypes.c_void_p) for p in arg_ptrs])
            times.append(t)
        print(median(times))

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


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


if __name__ == "__main__":
    main()
