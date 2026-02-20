"""Minimal SO runner test for debugging SIGSEGV."""
import ctypes
import os
import subprocess
import sys
import tempfile
import signal
import faulthandler

faulthandler.enable()

import numpy as np

from mlir._mlir_libs._mlir.ir import Context, Module, MemRefType, IntegerType, F64Type
from mlir.runtime import get_ranked_memref_descriptor
from utils import bufferize, lower
from mlir.dialects.func import FuncOp


def main():
    code = sys.stdin.read()

    with Context() as ctx:
        ctx.load_all_available_dialects()
        module = Module.parse(code)

    bufferize(module)

    # Get shapes from module
    main_func = next(op for op in module.body.operations if isinstance(op, FuncOp) and (op.name.value == 'main'))
    inputs = []
    outputs = []
    for input_type, input_attrs in zip(main_func.type.inputs, main_func.arg_attrs):
        assert isinstance(input_type, MemRefType)
        if "bufferize.result" in input_attrs:
            outputs.append(np.empty(input_type.shape, dtype=np.float64))
        else:
            inputs.append(np.full(input_type.shape, 2, dtype=np.float64))

    print(f"Input shapes: {[x.shape for x in inputs]}", file=sys.stderr)
    print(f"Output shapes: {[x.shape for x in outputs]}", file=sys.stderr)

    lower(module, sys.argv[1])

    # Compile to shared library
    tmpdir = tempfile.mkdtemp()
    mlir_file = os.path.join(tmpdir, 'input.mlir')
    ll_file = os.path.join(tmpdir, 'input.ll')
    opt_file = os.path.join(tmpdir, 'opt.ll')
    obj_file = os.path.join(tmpdir, 'output.o')
    so_file = os.path.join(tmpdir, 'output.so')

    with open(mlir_file, 'w') as f:
        f.write(str(module.operation))

    print("Translating to LLVM IR...", file=sys.stderr)
    r = subprocess.run(['mlir-translate', '-mlir-to-llvmir', mlir_file, '-o', ll_file], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"mlir-translate failed: {r.stderr}", file=sys.stderr)
        sys.exit(1)

    print("Running opt -O3...", file=sys.stderr)
    r = subprocess.run(['opt', '--passes=default<O3>', ll_file, '-S', '-o', opt_file], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"opt failed: {r.stderr}", file=sys.stderr)
        sys.exit(1)

    # Check the function signature in optimized IR
    with open(opt_file) as f:
        for line in f:
            if '_mlir_ciface_main' in line and 'define' in line:
                print(f"Function sig after opt: {line.strip()}", file=sys.stderr)

    print("Running llc...", file=sys.stderr)
    r = subprocess.run(['llc', '-O3', '-mcpu=broadwell', '-relocation-model=pic', '-filetype=obj', opt_file, '-o', obj_file], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"llc failed: {r.stderr}", file=sys.stderr)
        sys.exit(1)

    print("Linking...", file=sys.stderr)
    r = subprocess.run(['gcc', '-shared', '-o', so_file, obj_file,
        '-L/home/mt5383/.conda/envs/main/lib',
        '-lmlir_runner_utils', '-lmlir_c_runner_utils', '-lomp',
        '-Wl,-rpath,/home/mt5383/.conda/envs/main/lib'], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"gcc link failed: {r.stderr}", file=sys.stderr)
        sys.exit(1)

    # Check exported symbols
    r = subprocess.run(['nm', '-D', so_file], capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if 'main' in line.lower():
            print(f"Symbol: {line}", file=sys.stderr)

    print("Loading shared library...", file=sys.stderr)
    runner_utils = ctypes.CDLL("/home/mt5383/.conda/envs/main/lib/libmlir_runner_utils.so")
    c_runner_utils = ctypes.CDLL("/home/mt5383/.conda/envs/main/lib/libmlir_c_runner_utils.so")
    omp_lib = ctypes.CDLL("/home/mt5383/.conda/envs/main/lib/libomp.so")
    lib = ctypes.CDLL(so_file)

    func = lib._mlir_ciface_main
    func.restype = ctypes.c_int64

    # Build args: 3 memref descriptors (A, B, C)
    args = [
        ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arr)))
        for arr in inputs + outputs
    ]
    print(f"Number of args: {len(args)}", file=sys.stderr)
    print(f"Arg types: {[type(a) for a in args]}", file=sys.stderr)

    print("Calling function...", file=sys.stderr)
    sys.stderr.flush()
    result = func(*args)
    print(f"Result: {result}", file=sys.stderr)

    # Verify correctness
    expected = np.matmul(inputs[0], inputs[1])
    np.testing.assert_allclose(outputs[0], expected)
    print("Correctness verified!", file=sys.stderr)
    print(result)

    # Cleanup
    for f_path in [mlir_file, ll_file, opt_file, obj_file, so_file]:
        os.unlink(f_path)
    os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
