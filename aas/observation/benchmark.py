from dataclasses import dataclass
import os
import subprocess
from aas.observation.operation import OperationFeatures, NestedLoopFeatures, get_operation_type
from aas.action import Action
import re


@dataclass
class BenchmarkFeatures:
    """Dataclass to store the benchmark features data."""
    bench_name: str
    """The benchmark's name."""
    code: str
    """The MLIR code of the benchmark."""
    operation_tags: list[str]
    """List of operation tags."""
    operations: dict[str, OperationFeatures]
    """List of operations where each operation is represented by the OperationFeatures dataclass."""
    root_exec_time: int
    """Execution time of the benchmark in nanoseconds without any transformation."""

    def any_schedule_to_str(schedule: list[list[Action]]):
        """Convert a schedule to a string.

        Args:
            schedule (list[list[Action]]): The schedule to convert.

        Returns:
            str: The schedule as a string.
        """
        return '|'.join(''.join(str(op_action) for op_action in op_schedule) for op_schedule in schedule)


def extract_bench_features_from_code(bench_name: str, code: str, root_execution_time: int):
    """Extract benchmark features from the given code.

    Args:
        bench_name (str): the benchmark name
        code (str): the code to extract features from
        root_execution_time (int): the root execution time

    Returns:
        BenchmarkFeatures: the extracted benchmark features
    """
    result = subprocess.run(
        f'{os.getenv("AST_DUMPER_BIN_PATH")} -',
        shell=True,
        input=code.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    raw_ast_info = result.stdout.decode('utf-8')

    return __extract_bench_features_from_ast_result(bench_name, raw_ast_info, root_execution_time)


def extract_bench_features_from_file(bench_name: str, file_path: str, root_execution_time: int):
    """Extract benchmark features from the code in the file.

    Args:
        bench_name (str): the benchmark name
        file_path (str): the file path
        root_execution_time (int): the root execution time

    Returns:
        BenchmarkFeatures: the extracted benchmark features
    """
    result = subprocess.run(
        f'{os.getenv("AST_DUMPER_BIN_PATH")} {file_path}',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    raw_ast_info = result.stdout.decode('utf-8')

    return __extract_bench_features_from_ast_result(bench_name, raw_ast_info, root_execution_time)


def __extract_bench_features_from_ast_result(bench_name: str, raw_ast_info: str, root_execution_time: int):
    """Extracts benchmark features from the code's AST result and execution time.

    Args:
        bench_name (str): the benchmark name
        raw_ast_info (str): the raw AST information
        root_execution_time (int): the root execution time
        execution_time (int): the execution time

    Returns:
        BenchmarkFeatures: extracted benchmark features
    """
    info, full_code = raw_ast_info.split("########################################")
    operations_lines, _ = info.split('#BEGIN_GRAPH')

    operations_blocks = operations_lines.split('#START_OPERATION')
    operations_blocks = [block.strip() for block in operations_blocks if block]

    ops_tags = []
    operations = {}
    for operation_block in operations_blocks:
        raw_operation, rest = operation_block.split("#START_VECTORIZABLE")
        operation_type = get_operation_type(raw_operation)
        if operation_type is None:
            continue

        nested_loops = []
        op_count = {}
        load_data: list[list[str]] = []
        store_data: list[str] = []

        vectorizable_str, rest = rest.split("#START_NESTED_LOOPS")
        assert vectorizable_str.strip() in ["true", "false"], f"Vectorizable string is not valid: {vectorizable_str}"
        vectorizable = vectorizable_str.strip() == "true"

        nested_loops_str, rest = rest.split("#START_LOAD_DATA")
        op_iter_space_size = 1
        for nested_loop_str in nested_loops_str.strip().split("\n"):
            if not nested_loop_str:
                continue
            arg, low, high, step, iter = nested_loop_str.strip().split(" ")
            nested_loop = NestedLoopFeatures(
                arg=f'%{arg}',
                lower_bound=int(low),
                upper_bound=int(high),
                step=int(step),
                iterator_type=iter
            )
            op_iter_space_size *= nested_loop.upper_bound - nested_loop.lower_bound
            nested_loops.append(nested_loop)

        loads_data_str, rest = rest.split("#START_STORE_DATA")
        loads_data_str = re.sub(r'd\d+', lambda m: f'%{m.group()}', loads_data_str)
        for load_data_str in loads_data_str.strip().split("\n"):
            if not load_data_str:
                continue
            load_data.append(load_data_str.split(", "))

        store_data_str, rest = rest.split("#START_OP_COUNT")
        store_data_str = re.sub(r'd\d+', lambda m: f'%{m.group()}', store_data_str)
        store_data_list = store_data_str.strip().split("\n")
        assert len(store_data_list) == 1, f"Store data list is not of length 1: {store_data_list}"
        store_data = store_data_list[0].split(", ")

        ops_count_str, rest = rest.split("#START_TAG")
        for op_count_str in ops_count_str.strip().split("\n"):
            op, count = op_count_str.strip().split(" ")
            op_count[op] = int(count)

        operation_tag = rest.strip().split("\n")[0]
        ops_tags.append(operation_tag)
        operations[operation_tag] = OperationFeatures(
            operation_type=operation_type,
            op_count=op_count,
            op_iter_space_size=op_iter_space_size,
            nested_loops=nested_loops,
            load_data=load_data,
            store_data=store_data,
            vectorizable=vectorizable
        )

    return BenchmarkFeatures(
        bench_name=bench_name,
        code=full_code,
        operation_tags=ops_tags,
        operations=operations,
        root_exec_time=root_execution_time,
    )
