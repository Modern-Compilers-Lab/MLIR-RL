from utils.config import Config
from .base import Action
from rl_autoschedular.transforms import transform_img2col
from rl_autoschedular.state import OperationFeatures, OperationState, OperationType
from typing import Callable, Optional


class Img2Col(Action):
    """Class representing Img2Col action"""
    
    symbol = 'I2C'
    parameters: None

    def __init__(
        self,
        state: Optional[OperationState] = None,
        **extras
    ):
        super().__init__(
            state,
            **extras
        )

    def _apply_ready(self, code):
        original_code = code
        try:
            return transform_img2col(code, self.operation_tag)
        except Exception:
            return original_code
