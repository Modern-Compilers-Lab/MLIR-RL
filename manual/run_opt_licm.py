"""Run MLIR matmul with targeted LICM pre-processing (no full O3).

Pipeline: MLIR → lower → mlir-translate → opt (SROA+instcombine+LICM only) → reimport → JIT(opt_level=3)

This promotes C accumulator loads/stores out of the K-loop without the full O3
pipeline that may regress performance on some matmul sizes.
"""

import argparse
import ctypes
import ctypes.util
import os
import subprocess
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

    # Get MLIR LLVM dialect text
    asm = module.operation.get_asm()

    # Pipeline: MLIR → LLVM IR → targeted LICM → reimport → JIT
    with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as f:
        f.write(asm)
        mlir_file = f.name

    with tempfile.NamedTemporaryFile(suffix='.ll', mode='w', delete=False) as f:
        ll_file = f.name

    with tempfile.NamedTemporaryFile(suffix='.ll', mode='w', delete=False) as f:
        opt_file = f.name

    with tempfile.NamedTemporaryFile(suffix='.mlir', mode='w', delete=False) as f:
        reimport_file = f.name

    try:
        # Step 1: MLIR to LLVM IR
        result = subprocess.run(
            ['mlir-translate', '-mlir-to-llvmir', mlir_file, '-o', ll_file],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'mlir-translate failed: {result.stderr}')

        # Step 2: Targeted LICM passes only (promotes C accumulators to registers)
        # sroa: Scalar Replacement of Aggregates
        # instcombine: Instruction combining (simplifies load/store patterns)
        # licm: Loop Invariant Code Motion (promotes C loads/stores out of K-loop)
        passes = os.environ.get('OPT_PASSES', 'function(sroa,instcombine,loop-simplify,lcssa,loop-mssa(licm))')
        opt_cmd = ['opt', f'--passes={passes}']
        llvm_opts = os.environ.get('LLVM_OPTS', '').split()
        opt_cmd.extend(llvm_opts)
        opt_cmd.extend([ll_file, '-S', '-o', opt_file])
        result = subprocess.run(
            opt_cmd,
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'opt failed: {result.stderr}')

        # Step 3: Import back to MLIR
        result = subprocess.run(
            ['mlir-translate', '-import-llvm', opt_file, '-o', reimport_file],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'mlir-translate import failed: {result.stderr}')

        # Step 4: Parse optimized MLIR
        with open(reimport_file) as f:
            opt_mlir = f.read()

        with Context() as ctx2:
            ctx2.load_all_available_dialects()
            module = Module.parse(opt_mlir)

    finally:
        for f in [mlir_file, ll_file, opt_file, reimport_file]:
            try:
                os.unlink(f)
            except OSError:
                pass

    import sys
    print('targeted LICM pipeline complete', file=sys.stderr)

    # JIT opt_level from env (default 3)
    jit_opt = int(os.environ.get('JIT_OPT_LEVEL', '3'))
    execution_engine = ExecutionEngine(
        module,
        opt_level=jit_opt,
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
