from pathlib import Path
import ctypes
import argparse
from statistics import median
import numpy as np
import torch

from mlir.ir import Context, Module, MemRefType, IntegerType, F64Type, F32Type
from mlir.runtime import get_ranked_memref_descriptor
from mlir.dialects.func import FuncOp

from transformation import compile_aot, transform_module, bufferize_module, apply_pipeline_to_module

PARENT_DIR = Path(__file__).parents[2]


def transform_and_run(id: str, transform_schedule: str, mlir_passes: str, llvm_passes, llvm_flags: str, llc_flags: str, bufferize_first: bool):
    name, instance = id.rsplit("_", 1)
    with open(PARENT_DIR / 'data' / name / f'{instance}.mlir', 'r') as f:
        code = f.read()

    with Context() as ctx:
        ctx.load_all_available_dialects()
        module = Module.parse(code)

    if bufferize_first:
        bufferize_module(module)

    transform_module(module, transform_schedule)

    if not bufferize_first:
        bufferize_module(module)

    inputs, outputs = create_params(module)
    match name:
        case "matmul":
            expected = np.matmul(inputs[0], inputs[1])
        case "conv_2d":
            expected = torch.nn.functional.conv2d(
                torch.from_numpy(inputs[0]),
                torch.from_numpy(inputs[1])
            ).numpy()
        case _:
            raise ValueError(f"Unsupported benchmark name: {name}")
    args_list = convert_to_args(inputs, outputs)

    apply_pipeline_to_module(module, mlir_passes)

    times: list[int] = []
    with compile_aot(str(module), llvm_passes, llvm_flags, llc_flags) as func:
        func(*args_list)
        np.testing.assert_allclose(outputs[0], expected)

        for _ in range(10):
            func(*args_list)

        for _ in range(11):
            exec_time = func(*args_list)
            times.append(exec_time)
    return median(times)


def create_params(module: Module):
    """Creates the input and output parameters for the given MLIR module.

    Args:
        module: The MLIR module to create the parameters for.

    Returns:
        The list of inputs as numpy arrays
        The outputs structure (output arrays + delta)
    """
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

    # Get the main function
    main_func = next(op for op in module.body.operations if isinstance(op, FuncOp) and (op.name.value == 'main'))

    # Create input params
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

    # Create results arg
    res_types = main_func.type.results

    if not (len(res_types) == 1 and isinstance(res_types[0], IntegerType) and res_types[0].width == 64):
        raise Exception(f'unexpected result types {res_types}, expected a single i64 result for execution time')

    return inputs, outputs


def convert_to_args(inputs: list[np.ndarray], outputs: list[np.ndarray]) -> list:
    """Converts input arrays and output structure into ctypes arguments for MLIR execution.

    Prepares arguments in the format required by the MLIR execution engine. Each argument
    is a double pointer (pointer to pointer) to allow proper handling in the C calling
    convention.

    Args:
        inputs: List of input numpy arrays to be passed to the MLIR kernel.
        outputs_structure: ctypes Structure containing output memref descriptors and
            execution time.

    Returns:
        List of double pointers to ctypes Structures suitable for passing to ExecutionEngine.invoke().
    """
    args: list[ctypes._Pointer[ctypes.Structure]] = [
        ctypes.pointer(get_ranked_memref_descriptor(arr))
        for arr in inputs + outputs
    ]
    return args


def main():
    parser = argparse.ArgumentParser(description='Transform and run MLIR code with specified schedules and passes.')
    parser.add_argument('-i', '--id', type=str, required=True, help='The unique identifier for the MLIR code to transform. It takes the form "{name}_{instance}", where "name" is the name of the benchmark (e.g. "matmul") and "instance" is the specific instance (e.g. "0", "1", etc.).')
    parser.add_argument('-t', '--transform_schedule_file', type=str, required=True, help='The file containing the transformation schedule to apply.')
    parser.add_argument('-p', '--mlir_passes_file', type=str, required=True, help='The file containing the MLIR passes to apply during lowering.')
    parser.add_argument('--llvm_passes', type=str, default="default<O3>", help='LLVM opt pass pipeline to run before JIT (e.g. "licm,loop-unroll").')
    parser.add_argument('--llvm_flags', type=str, default="", help='Comma-separated LLVM CL flags (e.g. "enable-loop-versioning-licm,licm-mssa-max-acc-promotion=1000").')
    parser.add_argument('--llc_flags', type=str, default="", help='Comma-separated flags for llc codegen (e.g. "align-loops=32,enable-split-loopiv-heuristic").')
    parser.add_argument('--no-bufferize', action='store_true', help='Do not apply bufferization before applying the transformation schedule.')

    args = parser.parse_args()

    with open(args.transform_schedule_file, 'r') as f:
        transform_schedule = f.read()

    with open(args.mlir_passes_file, 'r') as f:
        mlir_passes = f.read()

    print(transform_and_run(args.id, transform_schedule, mlir_passes, args.llvm_passes, args.llvm_flags, args.llc_flags, not args.no_bufferize))


if __name__ == "__main__":
    main()
