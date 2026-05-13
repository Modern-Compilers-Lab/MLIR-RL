from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v0.implementation.name import Name

ACTION_CLASSES: list[type[ActionBase]] = [
    Name
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {}