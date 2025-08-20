from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from enum import Enum
from rl_autoschedular import config as cfg
import re
import os
import subprocess

from utils.log import print_error

if TYPE_CHECKING:
    from rl_autoschedular.actions.base import Action


class OperationType(Enum):
    Generic = 'generic'
    Matmul = 'matmul'
    Conv2D = 'conv_2d'
    Pooling = 'pooling'
    Add = 'add'

    unknown = ''


class IteratorType(Enum):
    Parallel = 'parallel'
    Reduction = 'reduction'


@dataclass
class NestedLoopFeatures:
    """Dataclass to store the nested loops features data."""
    arg: str
    """The argument representing the loop iterator."""
    lower_bound: int
    """The lower bound of the loop."""
    upper_bound: int
    """The upper bound of the loop."""
    step: int
    """The loop step."""
    iterator_type: IteratorType
    """The type of the loop iterator."""

    def copy(self):
        """Copy the current NestedLoopFeatures object."""
        return NestedLoopFeatures(self.arg, self.lower_bound, self.upper_bound, self.step, self.iterator_type)


@dataclass
class OperationFeatures:
    """Dataclass to store the operation features data."""
    raw_operation: str
    """The raw operation string without wrapping or transformations."""
    operation_type: OperationType
    """The type of the operation (generic, matmul, conv2d, ...)."""
    op_count: dict[str, int]
    """Number of arithmetic operations in the operation."""
    load_data: list[list[str]]
    """List of load accesses where each load is represented by the list of access arguments."""
    store_data: list[str]
    """List of store accesses where each store is represented by the list of access arguments."""
    nested_loops: list[NestedLoopFeatures]
    """List of nested loops where each loop is represented by the NestedLoopFeatures dataclass."""
    producers: list[str]
    """List of tags of operations that are consumed by the current operation"""
    vectorizable: bool
    """Flag to indicate if the operation is vectorizable."""

    def copy(self):
        """Copy the current OperationFeatures object."""
        return OperationFeatures(
            self.raw_operation,
            self.operation_type,
            self.op_count.copy(),
            [load.copy() for load in self.load_data],
            self.store_data.copy(),
            [loop.copy() for loop in self.nested_loops],
            self.producers.copy(),
            self.vectorizable
        )


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

    def copy(self):
        """Copy the current BenchmarkFeatures object."""
        return BenchmarkFeatures(
            self.bench_name,
            self.code,
            self.operation_tags.copy(),
            {tag: op.copy() for tag, op in self.operations.items()},
            self.root_exec_time
        )


@dataclass
class OperationState:
    bench_idx: int
    """The benchmark's index."""
    bench_name: str
    """The benchmark's name."""
    operation_tag: str
    """Tag used to identify the operation in the MLIR code."""
    original_operation_features: OperationFeatures
    """Features of the operation that will be kept always unchanged."""
    operation_features: OperationFeatures
    """Features of the operation."""
    producer_tag: Optional[str]
    """Tag that identifies the selected producer"""
    producer_features: Optional[OperationFeatures]
    """Features of the selected producer"""
    transformed_code: str
    """The operation string with wrapping and transformations."""
    step_count: int
    """The current step in the list of transformations applied to the operation."""
    transformation_history: list[list['Action']]
    """List of transformations with their parameters applied to the operation."""
    tmp_file: str
    """Temporary file to store the MLIR code."""
    terminal: bool
    """Flag that determines if the state is terminal"""

    def copy(self):
        """Copy the current OperationState object."""
        return OperationState(
            self.bench_idx,
            self.bench_name,
            self.operation_tag,
            self.original_operation_features.copy(),
            self.operation_features.copy(),
            self.producer_tag,
            self.producer_features.copy() if self.producer_features is not None else None,
            self.transformed_code,
            self.step_count,
            [seq.copy() for seq in self.transformation_history],
            self.tmp_file,
            self.terminal
        )


def extract_bench_features_from_code(bench_name: str, code: str, root_execution_time: int):
    """Extract benchmark features from the given code.

    Args:
        bench_name (str): the benchmark name
        code (str): the code to extract features from
        root_execution_time (int): the root execution time
        execution_time (int): the execution time

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
        execution_time (int): the execution time

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
    operations_lines, graph_str = info.split('#BEGIN_GRAPH')

    operations_blocks = operations_lines.split('#START_OPERATION')
    operations_blocks = [block.strip() for block in operations_blocks if block]

    ops_tags = []
    operations: dict[str, OperationFeatures] = {}
    for operation_block in operations_blocks:
        rest, operation_tag = operation_block.split("#START_TAG")
        operation_tag = operation_tag.strip().split("\n")[0]
        log_info = f"- Bench: {bench_name} - Operation: {operation_tag}"

        raw_operation, rest = rest.split("#START_VECTORIZABLE")
        operation_type = __get_operation_type(raw_operation)
        if operation_type is None:
            print_error(log_info)
            print_error("Unsupported operation type:", raw_operation.split("\n")[0])
            continue

        nested_loops = []
        op_count = {}
        load_data: list[list[str]] = []
        store_data: list[str] = []

        vectorizable_str, rest = rest.split("#START_NESTED_LOOPS")
        assert vectorizable_str.strip() in ["true", "false"], f"Vectorizable string is not valid: {vectorizable_str}"
        vectorizable = vectorizable_str.strip() == "true"

        nested_loops_str, rest = rest.split("#START_LOAD_DATA")
        for nested_loop_str in nested_loops_str.strip().split("\n"):
            if not nested_loop_str:
                continue
            arg, low, high, step, iter = nested_loop_str.strip().split(" ")
            nested_loops.append(NestedLoopFeatures(
                arg=f'%{arg}',
                lower_bound=int(low),
                upper_bound=int(high),
                step=int(step),
                iterator_type=IteratorType(iter)
            ))
        if len(nested_loops) > cfg.max_num_loops:
            print_error(log_info)
            print_error(f"Number of loops {len(nested_loops)} is not supported")
            continue

        loads_data_str, rest = rest.split("#START_STORE_DATA")
        loads_data_str = re.sub(r'd\d+', lambda m: f'%{m.group()}', loads_data_str)
        for load_data_str in loads_data_str.strip().split("\n"):
            if not load_data_str:
                continue
            load_data.append(load_data_str.split(", "))
        if any(len(load) > cfg.max_num_load_store_dim for load in load_data):
            print_error(log_info)
            print_error(f"Number of load dims {len(load_data[-1])} is not supported")
            continue
        if len(load_data) > cfg.max_num_stores_loads:
            # We ignore this overflow, because there are many cases with a huge number of loads
            load_data = load_data[:cfg.max_num_stores_loads]

        store_data_str, ops_count_str = rest.split("#START_OP_COUNT")
        store_data_str = re.sub(r'd\d+', lambda m: f'%{m.group()}', store_data_str)
        store_data_list = store_data_str.strip().split("\n")
        assert len(store_data_list) == 1, f"Store data list is not of length 1: {store_data_list}"
        store_data = store_data_list[0].split(", ")
        if len(store_data) > cfg.max_num_load_store_dim:
            print_error(log_info)
            print_error(f"Number of store dims {len(store_data)} is not supported")
            continue

        for op_count_str in ops_count_str.strip().split("\n"):
            op, count = op_count_str.strip().split(" ")
            op_count[op] = int(count)

        ops_tags.append(operation_tag)
        operations[operation_tag] = OperationFeatures(
            raw_operation=raw_operation,
            operation_type=operation_type,
            op_count=op_count,
            load_data=load_data,
            store_data=store_data,
            nested_loops=nested_loops,
            producers=[],
            vectorizable=vectorizable
        )

    # Extracte Producer/Consumer features
    graph_str = graph_str.replace("#END_GRAPH", "")
    graph_lines = [(line.split(' --> ')[0], line.split(' --> ')[1]) for line in graph_str.strip().split("\n") if line]

    for producer, consumer in graph_lines:
        operations[consumer].producers.append(producer)

    return BenchmarkFeatures(
        bench_name=bench_name,
        code=full_code,
        operation_tags=ops_tags,
        operations=operations,
        root_exec_time=root_execution_time,
    )


def __get_operation_type(raw_operation: str) -> Optional[OperationType]:
    """Get the operation type from the raw operation string.

    Args:
        raw_operation (str): The raw operation string.

    Returns:
        Optional[OperationType]: The operation type or None if not found.
    """
    for operation_type in OperationType:
        if operation_type.value and f'linalg.{operation_type.value}' in raw_operation:
            return operation_type
    return OperationType.unknown
