from typing import Optional
from rl_autoschedular.actions.tiling import Tiling
from rl_autoschedular.state import OperationFeatures, OperationState
from rl_autoschedular.transforms import transform_pack
from utils.config import Config


class ArrayPacking(Tiling):
    """Class representing ArrayPacking action"""

    symbol = 'AP'

    def __init__(
        self,
        parameters: list[int],
        state: Optional[OperationState] = None,
        /,
        **extras
    ):
        super().__init__(parameters, state, packed=True, **extras)

    def __str__(self):
        return f"{self.symbol}({','.join(map(str, self.parameters)) if self.extras['packed'] else False})"

    @classmethod
    def is_allowed(cls, state):
        return (len(state.operation_features.nested_loops) * 2) <= Config().max_num_loops

    def _apply_ready(self, code: str):
        # Special case: In packing, failures can happen
        # due to MLIR's preconditions, so we can ignore them
        try:
            return transform_pack(code, self.operation_tag, self.parameters)
        except Exception:
            self.extras['packed'] = False
            return code

    def update_features(self, operation_features: OperationFeatures):
        raise NotImplementedError
