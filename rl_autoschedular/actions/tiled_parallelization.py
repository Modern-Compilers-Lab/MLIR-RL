from .tiling import Tiling
from rl_autoschedular.transforms import transform_dialect_tile, transform_dialect_TP
from rl_autoschedular.state import OperationState, IteratorType


class TiledParallelization(Tiling):
    """Class representing Tiled Parallelization action"""

    symbol = 'TP'

    def _apply_ready(self, state: OperationState):
        parallel_params = [
            0 if state.operation_features.nested_loops[i].iterator_type == IteratorType.Reduction
            else param for i, param in enumerate(self.parameters)
        ]
        tiling_params = [
            param if state.operation_features.nested_loops[i].iterator_type == IteratorType.Reduction
            else 0 for i, param in enumerate(self.parameters)
        ]
        new_code = transform_dialect_TP(state.transformed_code, state.operation_tag, parallel_params, state.tmp_file)
        new_code = transform_dialect_tile(new_code, state.operation_tag, tiling_params, state.tmp_file)

        return new_code, bool(new_code)
