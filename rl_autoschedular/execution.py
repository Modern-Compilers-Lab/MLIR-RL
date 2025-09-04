import os
import numpy as np
from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
from mlir.passmanager import PassManager
from typing import Optional, overload
from rl_autoschedular.transforms import transform_bufferize_and_lower_v
from rl_autoschedular.actions import Action
from utils.bindings_process import BindingsProcess
from utils.singleton import Singleton
import json
import re


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

    def execute_code(self, code: str, bench_name: str, seq: list[list[Action]]) -> tuple[int, bool, bool]:
        """Evaluates the given MLIR code with a timeout.

        Args:
            state (OperationState): The operation state to evaluate.
            tmp_exec_data_file (str): The path to the temporary execution data file.

        Returns:
            Optional[float]: the execution time in seconds.
            Union[Exception, bool]: the assertion result.
            bool: flag for cache miss
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
        with open(self.exec_data_file, "r") as file:
            data: dict[str, dict[str, int]] = json.load(file)

        for bench_name, bench_data in new_data.items():
            if bench_name not in data:
                data[bench_name] = {}
            data[bench_name].update(bench_data)

        with open(self.exec_data_file, "w") as file:
            json.dump(data, file, indent=4)

    def get_code_cache_key(self, seq: list[list[Action]]) -> str:
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
            pm.run(module.operation)
            execution_engine = ExecutionEngine(
                module,
                opt_level=3,
                shared_libs=os.getenv("MLIR_SHARED_LIBS", "").split(","),
            )

            inputs = self.__create_inputs(code)

            args = []
            for input_arg in inputs:
                args.append(ctypes.pointer(ctypes.pointer(
                    get_ranked_memref_descriptor(input_arg)
                )))

            delta_arg = (ctypes.c_int64 * 1)(0)
            args.append(delta_arg)

            execution_engine.invoke("main", *args)
            execution_engine.invoke("main", *args)

            return delta_arg[0], True

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
        with open(self.exec_data_file, "r") as file:
            data: dict[str, dict[str, int]] = json.load(file)

        if bench_name in data and cache_key in data[bench_name]:
            return data[bench_name][cache_key]

        # No hit in both cache files
        return None

    def __create_inputs(self, code) -> list[np.ndarray]:
        # TODO: Probably could have done it better with module
        main_pattern = r"func.func @main\(([^)]+)\)"
        main_params = re.search(main_pattern, code).group(1)
        main_shapes = [arg.split(':')[1].strip() for arg in main_params.split(',')]

        inputs: list[np.ndarray] = []
        for shape in main_shapes:
            assert shape.startswith('memref<') or shape.startswith('tensor<'), f'unexpected shape {shape}'
            *np_shape, dtype = shape.replace('memref<', '').replace('tensor<', '').replace('>', '').split('x')
            assert dtype[0] in ['f', 'i'] and dtype[1:] in ['32', '64'], f'unexpected dtype {dtype}'
            match dtype[0]:
                case 'f':
                    match dtype[1:]:
                        case '32':
                            np_dtype = np.float32
                        case '64':
                            np_dtype = np.float64
                case 'i':
                    match dtype[1:]:
                        case '32':
                            np_dtype = np.int32
                        case '64':
                            np_dtype = np.int64
            np_shape = list(map(int, np_shape))
            # if len(np_shape) > 0:
            #     inputs.append((np.random.rand(*np_shape) * 100).astype(np_dtype))
            # else:
            #     inputs.append(np.array(np.random.rand() * 100, dtype=np_dtype))
            inputs.append(np.zeros(np_shape, dtype=np_dtype))

        return inputs
