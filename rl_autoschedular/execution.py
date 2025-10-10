import os
import ctypes
import ctypes.util
import numpy as np
from mlir.ir import Context, Module, MemRefType, IntegerType, F64Type, F32Type
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor, make_nd_memref_descriptor, as_ctype, ranked_memref_to_numpy
from mlir.passmanager import PassManager
from mlir.dialects.func import FuncOp
from typing import TYPE_CHECKING, Optional, overload
from rl_autoschedular.transforms import transform_bufferize_and_lower_v
from utils.bindings_process import BindingsProcess
from utils.singleton import Singleton
import json

if TYPE_CHECKING:
    from rl_autoschedular.actions import Action


class Execution(metaclass=Singleton):
    """Class that deals with code execution and cache management"""

    exec_data_file: str
    """Path to the local file where exec data is cached"""

    main_exec_data: Optional[dict[str, dict[str, int]]]
    """External exec data that was read at the beginning of training"""

    @overload
    def __init__(self):
        """Get already existing instance"""
        ...

    @overload
    def __init__(self, exec_data_file: str):
        """Initialize a new first instance without main exec data"""
        ...

    @overload
    def __init__(self, exec_data_file: str, main_exec_data: dict[str, dict[str, int]]):
        """Initialize a new first instance"""
        ...

    def __init__(self, exec_data_file: Optional[str] = None, main_exec_data: Optional[dict[str, dict[str, int]]] = None):
        if exec_data_file is None:
            raise Exception("No existing instance of class Execution has been found")

        self.exec_data_file = exec_data_file
        self.main_exec_data = main_exec_data

    def execute_code(self, code: str, bench_name: str, seq: list[list['Action']]) -> tuple[int, bool, bool]:
        """Evaluates the given MLIR code with a timeout.

        Args:
            state (OperationState): The operation state to evaluate.
            tmp_exec_data_file (str): The path to the temporary execution data file.

        Returns:
            tuple[int,bool,bool]: (execution time in nanoseconds, assertion result, cache miss flag)
        """
        code_cache_key = self.get_code_cache_key(seq)
        cache_exec_time = self.__check_execution_cache(bench_name, code_cache_key)
        if cache_exec_time is not None:
            return cache_exec_time, True, False

        bufferized_code = transform_bufferize_and_lower_v(code)
        real_exec_time, success = self.__execute_bufferized_code(bufferized_code)
        return real_exec_time, success, True

    def update_execution_cache(self, new_data: dict[str, dict[str, int]]):
        """Update the temp execution cache with the new data.

        Args:
            new_data (dict[str, dict[str, int]]): The new data to update.
            tmp_exec_data_file (str): The path to the temporary execution data file.
        """
        if not self.exec_data_file:
            raise Exception("Execution data file not provided")

        with open(self.exec_data_file, "r") as file:
            data: dict[str, dict[str, int]] = json.load(file)

        for bench_name, bench_data in new_data.items():
            if bench_name not in data:
                data[bench_name] = {}
            data[bench_name].update(bench_data)

        with open(self.exec_data_file, "w") as file:
            json.dump(data, file, indent=4)

    def get_code_cache_key(self, seq: list[list['Action']]) -> str:
        """Get the code cache key for the given operation state.

        Args:
            state (OperationState): The operation state to get the code cache key.
            bench_data (BenchmarkFeatures): The benchmark features data.

        Returns:
            str: the code cache key.
        """
        ops_codes = []
        for op_seq in seq:
            # TODO: There might be edge cases where part of a seq is invalid `env.py:301`
            ops_codes.append(''.join(map(str, op_seq)))

        return '|'.join(ops_codes)

    def __execute_bufferized_code(self, code: str) -> tuple[int, bool]:
        """Lowers and runs the given MLIR code using Python bindings, then returns the execution time and assertion
        result (if the executed code returns the correct result).

        Args:
            code (str): The MLIR code to run.

        Returns:
            Optional[float]: the execution time in seconds.
            bool: the assertion result.
        """

        def execute_bind_call():
            pass_pipeline = """builtin.module(
                canonicalize,
                buffer-deallocation-pipeline,
                convert-bufferization-to-memref,
                convert-linalg-to-loops,
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
                convert-func-to-llvm,
                convert-index-to-llvm,
                convert-arith-to-llvm,
                convert-cf-to-llvm,

                reconcile-unrealized-casts,
                canonicalize,
                cse
            )"""

            with Context():
                module = Module.parse(code)
                pm = PassManager.parse(pass_pipeline)

            inputs, outs_struct = self.__create_params(module)
            args = self.__convert_to_args(inputs, outs_struct)

            pm.run(module.operation)
            execution_engine = ExecutionEngine(
                module,
                opt_level=3,
                shared_libs=os.getenv("MLIR_SHARED_LIBS", "").split(","),
            )

            try:
                for _ in range(2):
                    execution_engine.invoke("main", *args)
                    # If output tensors are needed call `get_results` before `free_outputs`
                    outs_struct.free_outputs()
            finally:
                outs_struct.free_outputs()

            return outs_struct.delta, True

        return BindingsProcess.call(execute_bind_call, timeout=600)

    def __check_execution_cache(self, bench_name: str, cache_key: str) -> Optional[int]:
        """Check the execution cache for the given operation state.

        Args:
            bench_name (str): The benchmark name to check.
            cache_key (str): The cache key to check.
            tmp_exec_data_file (str): The path to the temporary execution data file.

        Returns:
            Optional[int]: the execution time in nanoseconds if the operation is found in the cache, otherwise None.
        """
        # Start by checking the main execution data
        if self.main_exec_data and bench_name in self.main_exec_data and cache_key in self.main_exec_data[bench_name]:
            return self.main_exec_data[bench_name][cache_key]

        # If no hit in the main cache file, check the temporary cache file
        if not self.exec_data_file:
            return None

        with open(self.exec_data_file, "r") as file:
            data: dict[str, dict[str, int]] = json.load(file)

        if bench_name in data and cache_key in data[bench_name]:
            return data[bench_name][cache_key]

        # No hit in both cache files
        return None

    @staticmethod
    def __create_params(module: Module):
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
                            Execution.free_pointer(address)
                            setattr(self, field_name, mem_desc_T())

        outputs_structure = OutputsStructure()
        for i, (field_name, field_type) in enumerate(out_fields):
            out_arg = field_type()
            setattr(outputs_structure, field_name, out_arg)

        return inputs, outputs_structure

    @staticmethod
    def __convert_to_args(inputs: list[np.ndarray], outputs_structure: ctypes.Structure):
        args: list[ctypes._Pointer[ctypes._Pointer[ctypes.Structure]]] = []
        args.append(ctypes.pointer(ctypes.pointer(outputs_structure)))
        for in_arr in inputs:
            args.append(ctypes.pointer(ctypes.pointer(
                get_ranked_memref_descriptor(in_arr)
            )))
        return args

    @staticmethod
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
