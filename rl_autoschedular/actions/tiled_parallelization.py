from .tiling import Tiling, Optional
from rl_autoschedular.transforms import transform_dialect_tile, transform_dialect_TP
from rl_autoschedular.state import OperationState, IteratorType


class TiledParallelization(Tiling):
    """Class representing Tiled Parallelization action"""

    symbol = 'TP'
    parallel_params: list[int]
    tiling_params: list[int]

    def __init__(self, parameters: list[int], state: Optional[OperationState] = None):
        super().__init__(parameters, state)
        self.parallel_params = [
            0 if state.operation_features.nested_loops[i].iterator_type == IteratorType.Reduction
            else param for i, param in enumerate(self.parameters)
        ]
        self.tiling_params = [
            param if state.operation_features.nested_loops[i].iterator_type == IteratorType.Reduction
            else 0 for i, param in enumerate(self.parameters)
        ]

    def _apply_ready(self, state: OperationState):
        new_code = transform_dialect_TP(state.transformed_code, state.operation_tag, self.parallel_params, state.tmp_file)
        new_code = transform_dialect_tile(new_code, state.operation_tag, self.tiling_params, state.tmp_file)

        return new_code, bool(new_code)
