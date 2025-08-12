from .base import Action


class NoTransformation(Action):
    """Class representing No Transformation"""

    symbol = 'NT'
    parameters: None
    terminal = True

    def __init__(self, *_):
        super().__init__()

    def _apply_ready(self, state):
        return state.transformed_code, True
