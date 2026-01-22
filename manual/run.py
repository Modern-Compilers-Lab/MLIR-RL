import sys
import ctypes
import ctypes.util
from statistics import median
import numpy as np
from mlir._mlir_libs._mlir.ir import Context, Module, MemRefType, IntegerType, F64Type, F32Type  # type: ignore
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor, make_nd_memref_descriptor, as_ctype, ranked_memref_to_numpy
from mlir.passmanager import PassManager
from mlir.dialects.func import FuncOp
from mlir.dialects.transform import interpreter
from typing import Optional


def main():
    pass_pipeline = """builtin.module(
        canonicalize,
        buffer-deallocation-pipeline,
        convert-bufferization-to-memref,
        convert-linalg-to-loops,
        loop-invariant-code-motion,
        scf-forall-to-parallel,
        convert-scf-to-openmp,
        expand-strided-metadata,
        finalize-memref-to-llvm,
        convert-scf-to-cf,
        lower-affine,

        convert-openmp-to-llvm,
        convert-vector-to-llvm,
        convert-math-to-llvm,
        convert-math-to-libm,
        finalize-memref-to-llvm,
        mem2reg,
        convert-func-to-llvm,
        convert-index-to-llvm,
        convert-arith-to-llvm,
        convert-cf-to-llvm,

        reconcile-unrealized-casts,
        canonicalize,
        cse
    )"""

    code = sys.stdin.read()
    if not code:
        with open(sys.argv[1], 'r') as f:
            code = f.read()

    with Context():
        module = Module.parse(code)
        pm = PassManager.parse(pass_pipeline)

    bufferize(module)

    inputs, outs_struct = create_params(module)
    args = convert_to_args(inputs, outs_struct)

    pm.run(module.operation)

    execution_engine = ExecutionEngine(
        module,
        opt_level=3,
        shared_libs=[
            "/scratch/mt5383/llvm-project/build/lib//libmlir_runner_utils.so",
            "/scratch/mt5383/llvm-project/build/lib//libmlir_c_runner_utils.so",
            "/home/mt5383/.conda/envs/main/lib/libomp.so"
        ],
    )

    try:
        for _ in range(10):
            execution_engine.invoke("main", *args)
            outs_struct.free_outputs()

        times = []
        for _ in range(11):
            execution_engine.invoke("main", *args)
            outs_struct.free_outputs()
            times.append(outs_struct.delta)
        print(median(times))
    finally:
        outs_struct.free_outputs()


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
    for input_type in main_func.type.inputs:
        assert isinstance(input_type, MemRefType), f'unexpected input type {input_type}'
        # in_arr = np.zeros(input_type.shape, dtype=get_dtype(input_type))
        # in_arr = np.random.rand(*input_type.shape).astype(get_dtype(input_type))
        # Array filled with 2
        in_arr = np.full(input_type.shape, 2, dtype=get_dtype(input_type))
        inputs.append(in_arr)

    # Create results arg
    res_types = main_func.type.results

    exec_time_type = res_types[-1]
    if not (isinstance(exec_time_type, IntegerType) and exec_time_type.width == 64):
        raise Exception(f'unexpected exec time type {exec_time_type}')

    out_fields: list[tuple[str, type[ctypes.Structure]]] = []
    for i, out_type in enumerate(res_types[:-1]):
        assert isinstance(out_type, MemRefType), f'unexpected output type {out_type}'
        descriptor_type = make_nd_memref_descriptor(out_type.rank, as_ctype(get_dtype(out_type)))
        out_fields.append((f'out_{i}', descriptor_type))

    class _OutputsStructure(ctypes.Structure):
        _fields_ = [
            *out_fields,
            ("delta", ctypes.c_int64)
        ]
        delta: int

        def get_results(self):
            res: list[np.ndarray] = []
            for field_name, _ in out_fields:
                out_array = ranked_memref_to_numpy([getattr(self, field_name)])
                res.append(out_array.copy())
            return res

        def free_outputs(self):
            for field_name, mem_desc_T in out_fields:
                memref_descriptor: ctypes.Structure = getattr(self, field_name)
                allocated_ptr: Optional[ctypes.c_longlong] = getattr(memref_descriptor, 'allocated', None)

                if allocated_ptr:
                    address = ctypes.cast(allocated_ptr, ctypes.c_void_p)
                    if address.value:
                        free_pointer(address)
                        setattr(self, field_name, mem_desc_T())

    outputs_structure = _OutputsStructure()
    for i, (field_name, field_type) in enumerate(out_fields):
        out_arg = field_type()
        setattr(outputs_structure, field_name, out_arg)

    return inputs, outputs_structure


def convert_to_args(inputs: list[np.ndarray], outputs_structure) -> list:
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
    args: list[ctypes._Pointer[ctypes._Pointer[ctypes.Structure]]] = []
    args.append(ctypes.pointer(ctypes.pointer(outputs_structure)))
    for in_arr in inputs:
        args.append(ctypes.pointer(ctypes.pointer(
            get_ranked_memref_descriptor(in_arr)
        )))
    return args


def free_pointer(ptr: ctypes.c_void_p):
    """Free the memory pointed to by the given pointer using the C standard library.

    Args:
        ptr: The pointer to free.
    """
    # Find the C standard library
    libc_path = ctypes.util.find_library('c')
    if not libc_path:
        raise RuntimeError("C standard library not found.")
    libc = ctypes.CDLL(libc_path)

    # Define the signature for free
    free = libc.free
    free.argtypes = [ctypes.c_void_p]
    free.restype = None

    # Call free
    free(ptr)


def bufferize(module: Module):
    """Apply bufferization

    Args:
        module: The MLIR module to transform.
    """
    transform_code = """
    module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
            transform.structured.eliminate_empty_tensors %arg0 : !transform.any_op
            %empty = transform.structured.match ops{["tensor.empty"]} in %arg0 : (!transform.any_op) -> !transform.op<"tensor.empty">
            transform.bufferization.empty_tensor_to_alloc_tensor %empty : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">

            transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %arg0 {bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op

            transform.yield
        }
    }"""

    t_module = Module.parse(transform_code, module.context)
    interpreter.apply_named_sequence(module, t_module.body.operations[0], t_module)


if __name__ == "__main__":
    main()
