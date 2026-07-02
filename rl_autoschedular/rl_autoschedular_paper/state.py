"""State representation and feature extraction for MLIR operations.

This module provides data structures for representing benchmark and operation states,
including features like loops, memory accesses, and operation types. It also provides
functionality for extracting these features from MLIR AST using the AstDumper tool.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from enum import Enum
import re
import os
import subprocess

from rl_autoschedular_paper.utils.config import Config
from rl_autoschedular_paper.utils.log import print_error

if TYPE_CHECKING:
    from rl_autoschedular_paper.actions.base import Action


class OperationType(Enum):
    """Enumeration of operation types for MLIR operations.

    Attributes:
        Generic: Generic operation type.
        Matmul: Matrix multiplication operation type.
        Conv: Convolutional operation type.
        Pooling: Pooling operation type.
        Add: Add operation type.
        unknown: Unknown operation type.
    """
    Generic = 'generic'
    Matmul = 'matmul'
    Conv = 'conv'
    Pooling = 'pooling'
    Add = 'add'

    unknown = ''


class IteratorType(Enum):
    """Enumeration of iterator types for loop dimensions.

    Attributes:
        Parallel: Parallel iterator type.
        Reduction: Reduction iterator type.
    """
    Parallel = 'parallel'
    Reduction = 'reduction'


@dataclass
class NestedLoopFeatures:
    """Dataclass to store the nested loops features data.

    Attributes:
        arg: The argument representing the loop iterator.
        lower_bound: The lower bound of the loop.
        upper_bound: The upper bound of the loop.
        step: The loop step.
        iterator_type: The type of the loop iterator.
    """

    arg: str
    lower_bound: int
    upper_bound: int
    step: int
    iterator_type: IteratorType

    def copy(self) -> 'NestedLoopFeatures':
        """Copy the current [NestedLoopFeatures][..] object.

        Returns:
            The copy.
        """
        return NestedLoopFeatures(self.arg, self.lower_bound, self.upper_bound, self.step, self.iterator_type)


@dataclass
class OperationFeatures:
    """Dataclass to store the operation features data.

    Attributes:
        operation_name: The name of the mlir operation.
        operation_type: The type of the operation.
        op_count: Number of arithmetic operations in the operation.
        load_data: List of load accesses where each load is represented by the list of access arguments.
        store_data: List of store accesses where each store is represented by the list of access arguments.
        nested_loops: List of nested loops where each loop is represented by the [NestedLoopFeatures][..NestedLoopFeatures] dataclass.
        producers: List of tags of operations that are consumed by the current operation along with their operand indices.
        consumers: List of tags of operations that consume the current operation.
        vectorizable: Flag to indicate if the operation is vectorizable.
        pre_actions: List actions that are already applied the current operatiom.
    """

    operation_name: str
    operation_type: OperationType
    op_count: dict[str, int]
    load_data: list[list[str]]
    store_data: list[list[str]]
    nested_loops: list[NestedLoopFeatures]
    producers: list[tuple[str, int]]
    consumers: list[tuple[str, int]]
    vectorizable: bool
    pre_actions: list['Action']

    def copy(self) -> 'OperationFeatures':
        """Copy the current [OperationFeatures][..] object.

        Returns:
            The copy.
        """
        return OperationFeatures(
            self.operation_name,
            self.operation_type,
            self.op_count.copy(),
            [load.copy() for load in self.load_data],
            [store.copy() for store in self.store_data],
            [loop.copy() for loop in self.nested_loops],
            self.producers.copy(),
            self.consumers.copy(),
            self.vectorizable,
            self.pre_actions.copy()
        )


@dataclass
class BenchmarkFeatures:
    """Dataclass to store the benchmark features data.

    Attributes:
        bench_name: The name of the benchmark.
        code: The MLIR code of the benchmark.
        operation_tags: List of operation tags.
        operations: List of operations where each operation is represented by the [OperationFeatures][..OperationFeatures] dataclass.
        root_exec_time: Execution time of the benchmark in nanoseconds without any transformation.
    """

    bench_name: str
    code: str
    operation_tags: list[str]
    operations: dict[str, OperationFeatures]
    root_exec_time: int

    def copy(self) -> 'BenchmarkFeatures':
        """Copy the current [BenchmarkFeatures][..] object.

        Returns:
            The copy.
        """
        return BenchmarkFeatures(
            self.bench_name,
            self.code,
            self.operation_tags.copy(),
            {tag: op.copy() for tag, op in self.operations.items()},
            self.root_exec_time
        )


@dataclass
class OperationState:
    """Dataclass to store the operation state data.

    Attributes:
        bench_idx: The index of the benchmark.
        bench_name: The name of the benchmark.
        operation_tag: The tag of the operation.
        original_operation_features: The features of the operation that will be kept always unchanged.
        operation_features: The features of the operation.
        producer_tag: The tag of the selected producer.
        producer_operand_idx: The index of the producer's operand.
        producer_features: The features of the selected producer.
        transformation_history: List of transformations with their parameters applied to the operation.
        terminal: Flag to indicate if the state is terminal.
    """

    bench_idx: int
    bench_name: str
    operation_tag: str
    original_operation_features: OperationFeatures
    operation_features: OperationFeatures
    producer_tag: Optional[str]
    producer_operand_idx: Optional[int]
    producer_features: Optional[OperationFeatures]
    transformation_history: list[list['Action']]
    terminal: bool

    @property
    def current_history(self) -> list['Action']:
        """Get the transformation sequence of the current operation being optimized.

        Returns:
            The transformation sequence.
        """
        return self.transformation_history[0]

    @property
    def step_count(self) -> int:
        """Get the number of steps in the current transformation sequence.

        Returns:
            The number of steps.
        """
        return len(self.current_history)

    @property
    def latest_action(self) -> Optional['Action']:
        """Get the latest action in the current transformation sequence.

        Returns:
            The latest action.
        """
        return self.current_history[-1] if self.current_history else None

    @property
    def has_incomplete_action(self) -> bool:
        """Check if the latest action is incomplete.

        Returns:
            True if the latest action is incomplete, False otherwise.
        """
        return (not self.latest_action.ready) if self.latest_action else False

    def record_action(self, action: 'Action'):
        """Record an action in the current transformation sequence.

        Note:
            If the latest action is incomplete, it will be replaced by the new action,
            and all the past incomplete actions will be kept in `sub_actions`.

        Args:
            action: The action to record.
        """
        if self.has_incomplete_action:
            # Case where the last action should be replaced
            action.sub_actions = self.latest_action.sub_actions + [self.latest_action]
            self.current_history[-1] = action
        else:
            self.current_history.append(action)

    def copy(self) -> 'OperationState':
        """Copy the current [OperationState][..] object.

        Returns:
            The copy.
        """
        return OperationState(
            self.bench_idx,
            self.bench_name,
            self.operation_tag,
            self.original_operation_features.copy(),
            self.operation_features.copy(),
            self.producer_tag,
            self.producer_operand_idx,
            self.producer_features.copy() if self.producer_features is not None else None,
            [seq.copy() for seq in self.transformation_history],
            self.terminal
        )


def extract_bench_features_from_code(bench_name: str, code: str, root_execution_time: int) -> BenchmarkFeatures:
    """Extract benchmark features from the given code.

    Args:
        bench_name: the benchmark name
        code: the code to extract features from
        root_execution_time: the root execution time

    Returns:
        the extracted benchmark features
    """
    result = subprocess.run(
        f'{os.getenv("AST_DUMPER_BIN_PATH")} -',
        shell=True,
        input=code.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    raw_ast_info = result.stdout.decode('utf-8')
    if result.returncode != 0:
        raise Exception(result.stderr.decode('utf-8'))

    return __extract_bench_features_from_ast_result(bench_name, raw_ast_info, root_execution_time)


def extract_bench_features_from_file(bench_name: str, file_path: str, root_execution_time: int) -> BenchmarkFeatures:
    """Extract benchmark features from the code in the file.

    Args:
        bench_name: the benchmark name
        file_path: the file path
        root_execution_time: the root execution time

    Returns:
        the extracted benchmark features
    """
    result = subprocess.run(
        f'{os.getenv("AST_DUMPER_BIN_PATH")} {file_path}',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    raw_ast_info = result.stdout.decode('utf-8')
    if result.returncode != 0:
        raise Exception(result.stderr.decode('utf-8'))

    return __extract_bench_features_from_ast_result(bench_name, raw_ast_info, root_execution_time)


def __extract_bench_features_from_ast_result(bench_name: str, raw_ast_info: str, root_execution_time: int) -> BenchmarkFeatures:
    """Extracts benchmark features from the code's AST result and execution time.

    Args:
        bench_name: the benchmark name
        raw_ast_info: the raw AST information
        root_execution_time: the root execution time

    Returns:
        extracted benchmark features
    """
    cfg = Config()

    info, full_code = raw_ast_info.split("########################################")
    operations_lines, graph_str = info.split('#BEGIN_GRAPH')

    operations_blocks = operations_lines.split('#START_OPERATION')
    operations_blocks = [block.strip() for block in operations_blocks if block]

    ops_tags = []
    operations: dict[str, OperationFeatures] = {}
    true_loads_count: dict[str, int] = {}
    for operation_block in operations_blocks:
        rest, operation_tag = operation_block.split("#START_TAG")
        operation_tag = operation_tag.strip().split("\n")[0]
        log_info = f"- Bench: {bench_name}\n- Operation: {operation_tag}"

        operation_name, rest = rest.split("#START_VECTORIZABLE")
        operation_name = operation_name.strip()
        operation_type = __get_operation_type(operation_name)

        nested_loops = []
        op_count = {}
        load_data: list[list[str]] = []
        store_data: list[list[str]] = []

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
            print_error(f"Number of loops {len(nested_loops)} is not supported\n" + log_info)
            continue

        loads_data_str, rest = rest.split("#START_STORE_DATA")
        loads_data_str = re.sub(r'd\d+', lambda m: f'%{m.group()}', loads_data_str)
        for load_data_str in loads_data_str.strip().split("\n"):
            if not load_data_str:
                continue
            load_data.append(load_data_str.split(", "))
        if any(len(load) > cfg.max_num_load_store_dim for load in load_data):
            print_error(f"Number of load dims {len(load_data[-1])} is not supported\n" + log_info)
            continue
        true_loads_count[operation_tag] = len(load_data)
        if len(load_data) > cfg.max_num_stores_loads:
            # We ignore this overflow, because there are many cases with a huge number of loads
            load_data = load_data[:cfg.max_num_stores_loads]

        stores_data_str, ops_count_str = rest.split("#START_OP_COUNT")
        stores_data_str = re.sub(r'd\d+', lambda m: f'%{m.group()}', stores_data_str)
        for store_data_str in stores_data_str.strip().split("\n"):
            if not store_data_str:
                continue
            store_data.append(store_data_str.split(", "))
        if any(len(store) > cfg.max_num_load_store_dim for store in store_data):
            print_error(f"Number of store dims {len(store_data[-1])} is not supported\n" + log_info)
            continue
        if len(store_data) > cfg.max_num_stores_loads:
            store_data = store_data[:cfg.max_num_stores_loads]

        for op_count_str in ops_count_str.strip().split("\n"):
            op, count = op_count_str.strip().split(" ")
            op_count[op] = int(count)

        ops_tags.append(operation_tag)
        operations[operation_tag] = OperationFeatures(
            operation_name=operation_name,
            operation_type=operation_type,
            op_count=op_count,
            load_data=load_data,
            store_data=store_data,
            nested_loops=nested_loops,
            producers=[],
            consumers=[],
            vectorizable=vectorizable,
            pre_actions=[]
        )

    # Extracte Producer/Consumer features
    graph_str = graph_str.replace("#END_GRAPH", "")
    graph_lines = [(line.split(' --> ')[0].split(' '), line.split(' --> ')[1].split(' ')) for line in graph_str.strip().split("\n") if line]

    for (producer, res_idx), (consumer, op_idx) in graph_lines:
        op_idx = int(op_idx)
        res_idx = int(res_idx)
        if op_idx >= len(operations[consumer].load_data):
            if 0 <= (op_idx - true_loads_count[consumer]) < len(operations[consumer].store_data):
                # Case where the index falls within the supported number of stores
                # -> align the index
                op_idx = op_idx - true_loads_count[consumer] + len(operations[consumer].load_data)
            else:
                # Case where the index falls within unsupported number of loads or stores
                # -> ignore
                continue

        operations[consumer].producers.append((producer, op_idx))
        operations[producer].consumers.append((consumer, res_idx))

    return BenchmarkFeatures(
        bench_name=bench_name,
        code=full_code,
        operation_tags=ops_tags,
        operations=operations,
        root_exec_time=root_execution_time,
    )


def __get_operation_type(operation_name: str) -> OperationType:
    """Get the operation type from the operation name.

    Args:
        operation_name: The operation name.

    Returns:
        The operation type or None if not found.
    """
    for operation_type in OperationType:
        if operation_type.value and operation_type.value in operation_name:
            return operation_type
    return OperationType.unknown
