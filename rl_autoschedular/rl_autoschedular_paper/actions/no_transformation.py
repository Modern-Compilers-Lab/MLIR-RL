"""No transformation action for MLIR loop transformations.

This module implements the no transformation action, which terminates the
transformation sequence without applying any changes.
"""

from typing import Optional
from rl_autoschedular_paper.state import OperationState
from .base import Action


class NoTransformation(Action):
    """Class representing No Transformation"""

    symbol = 'NT'

    parameters: None

    # --- constants ---
    terminal = True

    def __init__(self, state: Optional[OperationState] = None, /, **extras):
        super().__init__(state, **extras)

    def _apply_ready(self, module):
        return module
