from typing import Optional
import numpy as np
import ctypes.util

from mlir.ir import Context, Module, MemRefType, IntegerType, F64Type, F32Type
from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
from mlir.dialects.func import FuncOp
from mlir.runtime import get_ranked_memref_descriptor, make_nd_memref_descriptor, as_ctype, ranked_memref_to_numpy

from mlir.dialects.transform import interpreter
from utils.bindings_process import BindingsProcess

from llm_action.src.keys import MLIR_SHARED_LIBS
from llm_action.src.config import CODE_TRANSFORM_TIMEOUT, CODE_EXECUTION_TIMEOUT

def free_pointer(ptr: ctypes.c_void_p):
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

def convert_to_args(inputs: list[np.ndarray], outputs_structure: ctypes.Structure):
    args: list[ctypes._Pointer[ctypes._Pointer[ctypes.Structure]]] = []
    args.append(ctypes.pointer(ctypes.pointer(outputs_structure)))
    for in_arr in inputs:
        args.append(ctypes.pointer(ctypes.pointer(
            get_ranked_memref_descriptor(in_arr)
        )))
    return args

def create_params(module: Module):
    def __get_dtype(memref_type: MemRefType):
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
        in_arr = np.zeros(input_type.shape, dtype=__get_dtype(input_type))
        inputs.append(in_arr)

    # Create results arg
    res_types = main_func.type.results

    exec_time_type = res_types[-1]
    if not (isinstance(exec_time_type, IntegerType) and exec_time_type.width == 64):
        raise Exception(f'unexpected exec time type {exec_time_type}')

    out_fields: list[tuple[str, type[ctypes.Structure]]] = []
    for i, out_type in enumerate(res_types[:-1]):
        assert isinstance(out_type, MemRefType), f'unexpected output type {out_type}'
        descriptor_type = make_nd_memref_descriptor(out_type.rank, as_ctype(__get_dtype(out_type)))
        out_fields.append((f'out_{i}', descriptor_type))

    class OutputsStructure(ctypes.Structure):
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

    outputs_structure = OutputsStructure()
    for i, (field_name, field_type) in enumerate(out_fields):
        out_arg = field_type()
        setattr(outputs_structure, field_name, out_arg)

    return inputs, outputs_structure

def _transform_bind_call(code: str, transform_code: str) -> str:
    """Top-level function for subprocess isolation (must be picklable)."""
    try:
        from mlir.ir import Context, Module
        from mlir.dialects.transform import interpreter
        with Context():
            module = Module.parse(code)
            t_module = Module.parse(transform_code)
        interpreter.apply_named_sequence(module, t_module.body.operations[0], t_module)
        return str(module)
    except Exception as e:
        raise RuntimeError(str(e)) from None

def run_transform_code(code: str, transform_code: str, timeout: int = CODE_TRANSFORM_TIMEOUT) -> str:
    """Applies an MLIR transform sequence to the given code.

    Args:
        code (str): The MLIR code to transform.
        transform_code (str): The MLIR transform dialect code to apply.
        timeout (int, optional): Maximum time for transformation in seconds. Defaults to CODE_TRANSFORM_TIMEOUT.

    Returns:
        str: The transformed MLIR code as a string.
    """
    return BindingsProcess.call(_transform_bind_call, code, transform_code, timeout=timeout)

BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE = """
module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
        %all_loops = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
        transform.apply_licm to %all_loops : !transform.any_op

        transform.structured.eliminate_empty_tensors %arg0 : !transform.any_op
        %empty = transform.structured.match ops{["tensor.empty"]} in %arg0 : (!transform.any_op) -> !transform.op<"tensor.empty">
        transform.bufferization.empty_tensor_to_alloc_tensor %empty : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">

        %f0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        
        transform.apply_patterns to %f0 {
            transform.apply_patterns.vector.transfer_permutation_patterns
            transform.apply_patterns.vector.reduction_to_contract
        } : !transform.any_op
        transform.apply_patterns to %f0 {
            transform.apply_patterns.canonicalization
            transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
        } : !transform.any_op

        %arg1 = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %arg0 {bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op

        %f1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f1 {
            transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
            transform.apply_patterns.vector.transfer_permutation_patterns
            transform.apply_patterns.vector.lower_outerproduct
            transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"
            transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true
            transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
            transform.apply_patterns.vector.lower_shape_cast
            transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.yield
    }
}"""

def transform_bufferize_and_lower_v(code: str, transform_code: Optional[str] = None) -> str:
    """Apply the vectorization transformation with vectorizer to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.

    Returns:
        str: The code after applying the transformation.
    """
    if not transform_code:
        transform_code = BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE
    return run_transform_code(code, transform_code)

PASS_PIPELINE = [
    "canonicalize",
    "cse",
    "buffer-deallocation-pipeline",
    "convert-bufferization-to-memref",
    "convert-linalg-to-loops",
    "scf-forall-to-parallel",
    "convert-scf-to-openmp",
    "fold-memref-alias-ops",
    "expand-strided-metadata",
    "finalize-memref-to-llvm",
    "convert-scf-to-cf",
    "lower-affine",

    "convert-openmp-to-llvm",
    "convert-ub-to-llvm",
    "convert-vector-to-llvm{enable-x86vector}",
    "convert-math-to-llvm",
    "convert-math-to-libm",
    "finalize-memref-to-llvm",
    "convert-func-to-llvm",
    "convert-index-to-llvm",
    "arith-unsigned-when-equivalent",
    "convert-arith-to-llvm",
    "convert-cf-to-llvm",

    "reconcile-unrealized-casts",
    "canonicalize",
    "cse"
]

def _execute_bind_call(code: str, pass_pipeline_list: Optional[list[str]]) -> tuple[int, bool]:
    """Top-level function for subprocess isolation (must be picklable)."""
    try:
        import ctypes.util
        import numpy as np
        from mlir.ir import Context, Module
        from mlir.passmanager import PassManager
        from mlir.execution_engine import ExecutionEngine
        from llm_action.src.keys import MLIR_SHARED_LIBS

        if not pass_pipeline_list:
            execution_pass_pipeline = f"""builtin.module(
                {', '.join(PASS_PIPELINE)}
            )"""
        else:
            execution_pass_pipeline = "builtin.module(" + ", ".join(pass_pipeline_list) + ")"

        with Context():
            module = Module.parse(code)
            pm = PassManager.parse(execution_pass_pipeline)
        inputs, outs_struct = create_params(module)
        args = convert_to_args(inputs, outs_struct)

        pm.run(module.operation)
        execution_engine = ExecutionEngine(
            module,
            opt_level=3,
            shared_libs=MLIR_SHARED_LIBS.split(","),
        )

        try:
            for _ in range(2):
                execution_engine.invoke("main", *args)
                outs_struct.free_outputs()
        finally:
            outs_struct.free_outputs()

        return outs_struct.delta, True
    except Exception as e:
        # Capture the message string only and raise outside the except block.
        # Inside the except, Python implicitly chains the in-flight exception
        # to the new one via `__context__`; cloudpickle/pickle then traverses
        # __context__ during serialization and trips on MLIRError objects
        # (which carry unpicklable DiagnosticInfo). Detaching all chain
        # references before raising keeps the exception fully picklable.
        err_msg = str(e)
    sanitized = RuntimeError(err_msg)
    sanitized.__cause__ = None
    sanitized.__context__ = None
    sanitized.__suppress_context__ = True
    raise sanitized

def execute_bufferized_code(code: str, pass_pipeline: Optional[list[str]] = None, timeout: int = CODE_EXECUTION_TIMEOUT) -> tuple[int, bool]:
    """Lowers and runs the given MLIR code using Python bindings, then returns the execution time and assertion
    result (if the executed code returns the correct result).

    Args:
        code (str): The MLIR code to run.
        timeout (int): The maximum time to allow for code execution in seconds.

    Returns:
        int: the execution time in nanoseconds.
        bool: the assertion result.
    """
    return BindingsProcess.call(_execute_bind_call, code, pass_pipeline, timeout=timeout)


def _execute_from_path_bind_call(
    code_path: str,
    bufferize_transform_code: Optional[str],
    pass_pipeline_list: Optional[list[str]],
) -> tuple[int, bool]:
    """Spawn-child target for the Dask worker path.

    Reads the MLIR source from `code_path` *inside* the spawn child (skipping
    a multiprocessing-pipe pickle of the 10-100 KB code string), then runs the
    full bufferize→execute pipeline back-to-back. The bufferized intermediate
    stays in spawn-child memory — no second IPC round-trip. Errors are
    sanitized to plain `RuntimeError` so cloudpickle can transport them back
    through Dask without choking on `MLIRError`/`DiagnosticInfo`.
    """
    import re

    try:
        with open(code_path) as f:
            code = f.read()

        # Inline the _is_bufferized check from mlir_execution.py to avoid a
        # circular import (mlir_execution already imports from this module).
        params_match = re.search(r'func\.func @main\(([^)]+)\)', code)
        already_bufferized = bool(
            params_match
            and 'memref<' in params_match.group(1)
            and 'tensor<' not in params_match.group(1)
        )

        if already_bufferized:
            bufferized = code
        else:
            transform_code = bufferize_transform_code or BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE
            bufferized = _transform_bind_call(code, transform_code)

        return _execute_bind_call(bufferized, pass_pipeline_list)
    except Exception as e:
        # Same sanitization shape as _execute_bind_call: detach all chained
        # exception state so the resulting RuntimeError is fully picklable.
        err_msg = str(e)
    sanitized = RuntimeError(err_msg)
    sanitized.__cause__ = None
    sanitized.__context__ = None
    sanitized.__suppress_context__ = True
    raise sanitized


