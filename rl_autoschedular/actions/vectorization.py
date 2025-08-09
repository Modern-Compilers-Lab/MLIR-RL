from .base import Action
from rl_autoschedular import config as cfg
from rl_autoschedular.transforms import transform_dialect_vectorize, transform_dialect_tile, apply_conv2d_decomposition
from rl_autoschedular.state import OperationFeatures, OperationType


class Vectorization(Action):
    """Class representing Vectorization action"""

    symbol = 'V'
    parameters: None
    terminal = True

    def __init__(self):
        super().__init__()

    @classmethod
    def is_allowed(cls, state):
        is_pooling = state.operation_features.operation_type == OperationType.Pooling
        return cls.is_vectorizable(state.operation_features) or is_pooling

    def _apply_ready(self, state):
        is_legal = self.is_vectorizable(state.operation_features)

        code = state.transformed_code
        # Decompose pooling operation to make it vectorizable
        if not is_legal and state.operation_features.operation_type == OperationType.Pooling:
            # Tile the pooling operation for decomposition
            tile_sizes = []
            for i in range(len(state.operation_features.nested_loops)):
                if i in [2, 4]:
                    tile_sizes.append(1)
                else:
                    tile_sizes.append(0)
            new_code = transform_dialect_tile(code, state.operation_tag, tile_sizes, state.tmp_file)

            # Apply the decomposition
            new_code = apply_conv2d_decomposition(new_code, state.operation_tag, state.tmp_file)

            # Check if the decomposition was successful
            # and if so, replace the old code
            if new_code:
                is_legal = True
                code = new_code

        # Apply the vectorization if eligible
        if not is_legal:
            raise Exception("Operation is not vectorizable")
        new_code = transform_dialect_vectorize(code, state.operation_tag, state.tmp_file)

        return new_code, bool(new_code)

    @staticmethod
    def is_vectorizable(operation_features: OperationFeatures) -> bool:
        """Check if the operation is small enough for vectorization.

        Args:
            operation_features (OperationFeatures): The operation features.

        Returns:
            bool: Whether the operation is vectorizable or not.
        """
        if not operation_features.vectorizable:
            return False

        op_iter_space = 1
        for nested_loop in operation_features.nested_loops:
            op_iter_space *= nested_loop.upper_bound
        return op_iter_space <= cfg.vect_size_limit
