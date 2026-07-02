"""Tiled parallelization action for MLIR loop transformations.

This module implements the tiled parallelization transformation action, which applies
tiling with parallelization using forall constructs.
"""

from .tiling import Tiling
from rl_autoschedular_paper_transformer.transforms import transform_tile, transform_TP
from rl_autoschedular_paper_transformer.state import OperationState, IteratorType
from mlir._mlir_libs._mlir.ir import Module  # type: ignore
from typing import Optional


class TiledParallelization(Tiling):
    """Class representing Tiled Parallelization action"""

    symbol = 'TP'

    # --- extras ---
    parallel_params: list[int]
    tiling_params: list[int]

    def __init__(
        self,
        parameters: list[int],
        state: Optional[OperationState] = None,
        /, *,
        iterators: Optional[list[str]] = None,
        **extras
    ):
        if (state is None) == (iterators is None):
            raise ValueError("Either state or iterators must be provided and not both")
        if state:
            iterators = [loop.iterator_type.value for loop in state.operation_features.nested_loops]
        super().__init__(parameters, state, iterators=iterators, **extras)

        self.parallel_params = [
            0 if iterator == IteratorType.Reduction.value
            else param for param, iterator in zip(self.parameters, iterators)
        ]
        self.tiling_params = [
            param if iterator == IteratorType.Reduction.value
            else 0 for param, iterator in zip(self.parameters, iterators)
        ]

    @classmethod
    def is_allowed(cls, state):
        return not any(
            isinstance(action, Tiling) for action in
            state.operation_features.pre_actions + state.current_history
        )

    def _apply_ready(self, module: Module):
        transform_TP(module, self.operation_tag, self.parallel_params)
        transform_tile(module, self.operation_tag, self.tiling_params)
