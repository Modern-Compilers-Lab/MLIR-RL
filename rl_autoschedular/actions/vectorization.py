from .base import Action
from rl_autoschedular import config as cfg
from rl_autoschedular.transforms import transform_dialect_vectorize, transform_dialect_tile, apply_conv2d_decomposition
from rl_autoschedular.state import OperationState, OperationType


class Vectorization(Action):
    """Class representing Vectorization action"""

    symbol = 'V'
    parameters: None
    requires_decomposition: bool
    terminal = True

    def __init__(self, state: OperationState):
        super().__init__()
        self.requires_decomposition = False
        if state.operation_features.operation_type == OperationType.Pooling:
            assert len(state.operation_features.nested_loops) == 6
            self.requires_decomposition = True

    @classmethod
    def is_allowed(cls, state):
        if not state.operation_features.vectorizable:
            return False

        op_iter_space = 1
        for nested_loop in state.operation_features.nested_loops:
            op_iter_space *= nested_loop.upper_bound
        return op_iter_space <= cfg.vect_size_limit

    def _apply_ready(self, state):
        code = state.transformed_code

        # Decompose pooling operation to make it vectorizable
        if self.requires_decomposition:
            code, decomposed = self.__decompose_pooling(state)
            if not decomposed:
                raise Exception("Pooling decomposition not successful")

        new_code = transform_dialect_vectorize(code, state.operation_tag, state.tmp_file)

        return new_code, bool(new_code)

    @staticmethod
    def __decompose_pooling(state: OperationState) -> tuple[str, bool]:
        # Tile the pooling operation for decomposition
        tile_sizes = [0, 0, 1, 0, 1, 0]
        new_code = transform_dialect_tile(state.transformed_code, state.operation_tag, tile_sizes, state.tmp_file)

        # Apply the decomposition
        new_code = apply_conv2d_decomposition(new_code, state.operation_tag, state.tmp_file)

        return new_code, bool(new_code)
