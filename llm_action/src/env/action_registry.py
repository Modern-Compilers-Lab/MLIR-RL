import importlib
from dataclasses import dataclass

from llm_action.src.actions.base import ActionBase

@dataclass
class ActionRegistry:
    action_classes: list[type[ActionBase]]
    num_actions: int
    done_idx: int
    total_actions: int
    name_to_idx: dict[str, int]

def load_action_registry(version: str) -> ActionRegistry:
    mod = importlib.import_module(f"llm_action.src.actions.{version}.registry")
    classes = mod.ACTION_CLASSES
    n = len(classes)
    return ActionRegistry(
        action_classes=classes,
        num_actions=n,
        done_idx=n,
        total_actions=n + 1,
        name_to_idx={cls.__name__: i for i, cls in enumerate(classes)},
    )
