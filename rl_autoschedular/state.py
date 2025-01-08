from dataclasses import dataclass
from typing import Literal
import numpy as np


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
    iterator_type: Literal["parallel", "reduction"]
    """The type of the loop iterator."""


@dataclass
class OperationFeatures:
    """Dataclass to store the operation features data."""
    raw_operation: str
    """The raw operation string without wrapping or transformations."""
    op_count: dict[str, int]
    """Number of arithmetic operations in the operation."""
    load_data: list[list[str]]
    """List of load accesses where each load is represented by the list of access arguments."""
    store_data: list[list[str]]
    """List of store accesses where each store is represented by the list of access arguments."""
    nested_loops: list[NestedLoopFeatures]
    """List of nested loops where each loop is represented by the NestedLoopFeatures dataclass."""


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
    exec_time: int
    """Execution time of the benchmark in nanoseconds."""


@dataclass
class OperationState:
    bench_name: str
    """The benchmark's name."""
    operation_tag: str
    """Tag used to identify the operation in the MLIR code."""
    operation_type: str
    """The type of the operation (generic, matmul, conv2d, ...)."""
    operation_features: OperationFeatures
    """Features of the operation."""
    transformed_code: str
    """The operation string with wrapping and transformations."""
    actions: np.ndarray
    """Action parameters for parallelization, tiling and interchange. The shape is (MAX_NUM_LOOPS, 3, truncate)."""
    actions_mask: np.ndarray
    """Mask for the actions. The shape is (5 + L + L + (L-1) + (L-2) + (L-3)) where L = MAX_NUM_LOOPS."""
    step_count: int
    """The current step in the list of transformations applied to the operation."""
    exec_time: int
    """Execution time of the operation in nanoseconds."""
    root_exec_time: int
    """Execution time of the operation in nanoseconds without any transformation."""
    transformation_history: list[tuple[str, list[int]]]
    """List of transformations with their parameters applied to the operation."""
    cummulative_reward: float
    """Cummulative reward of the operation."""
    tmp_file: str
    """Temporary file to store the MLIR code."""
