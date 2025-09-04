from .tiling import Tiling, Optional
from rl_autoschedular.transforms import transform_tile, transform_TP
from rl_autoschedular.state import OperationState, IteratorType


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

    def _apply_ready(self, code: str):
        p_code = transform_TP(code, self.operation_tag, self.parallel_params)
        return transform_tile(p_code, self.operation_tag, self.tiling_params)
