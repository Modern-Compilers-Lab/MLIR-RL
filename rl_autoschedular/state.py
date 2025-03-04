from dataclasses import dataclass
from typing import Literal, Optional
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

    def copy(self):
        """Copy the current NestedLoopFeatures object."""
        return NestedLoopFeatures(self.arg, self.lower_bound, self.upper_bound, self.step, self.iterator_type)


@dataclass
class OperationFeatures:
    """Dataclass to store the operation features data."""
    raw_operation: str
    """The raw operation string without wrapping or transformations."""
    op_count: dict[str, int]
    """Number of arithmetic operations in the operation."""
    load_data: list[list[str]]
    """List of load accesses where each load is represented by the list of access arguments."""
    store_data: list[str]
    """List of store accesses where each store is represented by the list of access arguments."""
    nested_loops: list[NestedLoopFeatures]
    """List of nested loops where each loop is represented by the NestedLoopFeatures dataclass."""

    def copy(self):
        """Copy the current OperationFeatures object."""
        return OperationFeatures(
            self.raw_operation,
            self.op_count.copy(),
            [load.copy() for load in self.load_data],
            self.store_data.copy(),
            [loop.copy() for loop in self.nested_loops]
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


OperationType = Literal["generic", "matmul", "conv_2d", "pooling", "add"]


@dataclass
class OperationState:
    bench_name: str
    """The benchmark's name."""
    operation_tag: str
    """Tag used to identify the operation in the MLIR code."""
    operation_type: OperationType
    """The type of the operation (generic, matmul, conv2d, ...)."""
    operation_features: OperationFeatures
    """Features of the operation."""
    validated_code: str
    """The latest validated benchmark code (if not in inference, this will always be the original code)."""
    transformed_code: str
    """The operation string with wrapping and transformations."""
    actions: np.ndarray
    """Action parameters for parallelization, tiling and interchange. The shape is (truncate, 3, MAX_NUM_LOOPS)."""
    action_mask: np.ndarray
    """Mask for the actions. The shape is (5 + L + L) where L = MAX_NUM_LOOPS."""
    step_count: int
    """The current step in the list of transformations applied to the operation."""
    exec_time: int
    """Execution time of the operation in nanoseconds."""
    transformation_history: list[tuple[str, list[int]]]
    """List of transformations with their parameters applied to the operation."""
    interchange_permutation: list[int]
    """Current permutation of the interchange transformation (used only)."""
    tmp_file: str
    """Temporary file to store the MLIR code."""

    def copy(self):
        """Copy the current OperationState object."""
        return OperationState(
            self.bench_name,
            self.operation_tag,
            self.operation_type,
            self.operation_features.copy(),
            self.validated_code,
            self.transformed_code,
            self.actions.copy(),
            self.action_mask.copy(),
            self.step_count,
            self.exec_time,
            [(transformation, params.copy()) for transformation, params in self.transformation_history],
            self.interchange_permutation.copy(),
            self.tmp_file
        )

    def last_op_history_index(self) -> Optional[int]:
        """Get the index of the beginning of the history of the last operation."""
        history_len = len(self.transformation_history)
        if history_len == 0:
            return None
        if self.transformation_history[-1][0] == 'done':
            return None
        i = history_len - 1
        while i > 0 and self.transformation_history[i][0] != 'done':
            i -= 1
        if self.transformation_history[i][0] == 'done':
            return i + 1
        return i
