from .tiled_parallelization import TiledParallelization
from rl_autoschedular.transforms import transform_TF
from rl_autoschedular.state import OperationState
from typing import Optional


class TiledFusion(TiledParallelization):
    """Class representing Tiled Fusion action"""

    symbol = 'TPF'

    # --- extras ---
    producer_tag: str

    def __init__(
        self,
        parameters: list[int],
        state: Optional[OperationState] = None,
        producer_tag: Optional[str] = None,
        **extras
    ):
        if (state is None) == (producer_tag is None):
            raise ValueError("Either state or producer tag must be provided and not both")
        if state:
            producer_tag = state.producer_tag
        super().__init__(parameters, state, producer_tag=producer_tag, **extras)

        self.producer_tag = producer_tag

    def __str__(self):
        return f"{self.symbol}({self.producer_tag};{','.join(map(str, self.parameters))})"

    @classmethod
    def is_allowed(cls, state):
        already_fused = any(isinstance(action, cls) for action in state.transformation_history[0])
        has_producers = state.producer_tag is not None

        return has_producers and not already_fused

    def _apply_ready(self, code):
        return transform_TF(
            code,
            self.operation_tag,
            self.producer_tag,
            self.tiling_params,
            self.parallel_params,
        )
