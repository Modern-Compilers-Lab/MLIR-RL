"""Vectorization action for MLIR loop transformations.

This module implements the vectorization transformation action, which applies
vectorization to operations and handles preprocessing steps like transpose and decompose.
"""

from rl_autoschedular_paper.utils.config import Config
from .base import Action
from rl_autoschedular_paper.transforms import (
    move_module, transform_pre_vec, transform_vectorize, transform_tile,
    transform_decompose, transform_transpose_conv_2d
)
from rl_autoschedular_paper.state import OperationFeatures, OperationState, OperationType
from mlir._mlir_libs._mlir.ir import Module  # type: ignore
from typing import Callable, Optional


class Vectorization(Action):
    """Class representing Vectorization action"""

    symbol = 'V'
    parameters: None

    # --- constants ---
    terminal = True

    # --- extras ---
    preprocessing: list[Callable[[Module], None]]

    def __init__(
        self,
        state: Optional[OperationState] = None,
        /, *,
        requires_transpose: Optional[bool] = None,
        requires_decompose: Optional[bool] = None,
        decompose_tile_sizes: Optional[list[int]] = None,
        **extras
    ):
        args_is_none = [
            requires_transpose is None,
            requires_decompose is None,
            decompose_tile_sizes is None
        ]
        if (state is None) in args_is_none:
            raise ValueError("Either state or preprocessing attributes must be provided and not both")
        if state:
            op_feats = state.operation_features.copy()

            if op_feats.operation_type not in [OperationType.Pooling, OperationType.Conv]:
                requires_transpose, requires_decompose, decompose_tile_sizes = False, False, []
            else:
                if requires_transpose := self.__requires_transpose(op_feats):
                    op_feats.operation_name = 'linalg.conv_2d_nhwc_hwcf'
                decompose_tile_sizes = []
                if requires_decompose := self.__requires_decompose(op_feats):
                    decompose_tile_sizes = self.__decompose_tile_sizes(op_feats)
        super().__init__(
            state,
            requires_transpose=requires_transpose,
            requires_decompose=requires_decompose,
            decompose_tile_sizes=decompose_tile_sizes,
            vectorized=True,
            **extras
        )

        self.preprocessing = []
        if requires_transpose:
            self.preprocessing.append(lambda m: transform_transpose_conv_2d(m, self.operation_tag))
        if requires_decompose:
            self.preprocessing.append(lambda m: transform_tile(m, self.operation_tag, decompose_tile_sizes))
            self.preprocessing.append(lambda m: transform_decompose(m, self.operation_tag))
        self.preprocessing.append(lambda m: transform_pre_vec(m, self.operation_tag))

    def __str__(self):
        return f"{self.symbol}({self.extras['vectorized']})"

    @classmethod
    def is_allowed(cls, state):
        op_iter_space = 1
        for nested_loop in state.operation_features.nested_loops:
            op_iter_space *= nested_loop.upper_bound
        return op_iter_space <= Config().vect_size_limit

    def _apply_ready(self, module):
        module_clone: Module = module.operation.clone()
        # Special case: In vectorization failures can happen
        # due to MLIR's preconditions, so we can ignore them
        try:
            for pre in self.preprocessing:
                pre(module)
            transform_vectorize(module, self.operation_tag)
        except Exception:
            self.extras['vectorized'] = False
            move_module(module_clone, module)

    @classmethod
    def __requires_transpose(cls, operation_features: OperationFeatures) -> bool:
        return operation_features.operation_name == 'linalg.conv_2d_nhwc_fhwc'

    @classmethod
    def __requires_decompose(cls, operation_features: OperationFeatures) -> bool:
        """a.k.a is a two dimensional conv interface op"""

        if 'conv_2d' in operation_features.operation_name:
            return True

        if operation_features.operation_type == OperationType.Pooling and len(operation_features.nested_loops) >= 6:
            return True

        return False

    @classmethod
    def __decompose_tile_sizes(cls, operation_features: OperationFeatures) -> list[int]:
        tile_sizes = [0 for _ in operation_features.nested_loops]

        oh = None
        if operation_features.operation_name == 'linalg.conv_2d':
            oh = 0
        elif '_nhwc_' in operation_features.operation_name:
            oh = 1
        elif '_nchw_' in operation_features.operation_name:
            oh = 2

        kh = None
        if operation_features.operation_name == 'linalg.conv_2d':
            kh = 2
        elif '_fchw' in operation_features.operation_name:
            kh = 5
        elif '_hwc' in operation_features.operation_name or operation_features.operation_type == OperationType.Pooling:
            kh = 4

        if oh is not None and kh is not None:
            tile_sizes[oh] = 1
            tile_sizes[kh] = 1

        return tile_sizes
