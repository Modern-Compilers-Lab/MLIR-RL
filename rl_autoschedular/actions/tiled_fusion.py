from .tiled_parallelization import TiledParallelization
from rl_autoschedular.transforms import transform_dialect_TF
from rl_autoschedular.state import OperationState
from typing import Optional


class TiledFusion(TiledParallelization):
    """Class representing Tiled Fusion action"""

    symbol = 'TPF'
    producer_tag: str

    def __init__(self, parameters: list[int], state: Optional[OperationState] = None):
        if not state:
            raise ValueError("State is always required for fusion")
        self.producer_tag = state.producer_tag
        super().__init__(parameters, state)

    def __repr__(self):
        return f"{self.symbol}({self.producer_tag}|{','.join(map(str, self.parameters))})"

    @classmethod
    def is_allowed(cls, state):
        already_fused = any(isinstance(action, cls) for action in state.transformation_history[0])
        has_producers = state.producer_tag is not None

        return has_producers and not already_fused

    def _apply_ready(self, state: OperationState):
        new_code = transform_dialect_TF(
            state.transformed_code,
            state.operation_tag,
            self.producer_tag,
            self.tiling_params,
            self.parallel_params,
            state.tmp_file,
        )

        return new_code, bool(new_code)
