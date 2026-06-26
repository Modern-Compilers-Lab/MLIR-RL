import importlib
from dataclasses import dataclass, field

from llm_action.src.actions.base import ActionBase

@dataclass
class ActionRegistry:
    action_classes: list[type[ActionBase]]
    num_actions: int
    done_idx: int
    total_actions: int
    name_to_idx: dict[str, int]
    blocks: dict[int, frozenset[int]] = field(default_factory=dict)
    schedule_paths: dict[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)

def load_action_registry(version: str) -> ActionRegistry:
    mod = importlib.import_module(f"llm_action.src.actions.{version}.registry")
    classes = mod.ACTION_CLASSES
    n = len(classes)
    name_to_idx = {cls.__name__: i for i, cls in enumerate(classes)}

    raw_deps: dict[str, list[str]] = getattr(mod, "ACTION_DEPENDENCIES", {})
    blocks: dict[int, frozenset[int]] = {}
    for blocker_name, blocked_names in raw_deps.items():
        if blocker_name not in name_to_idx:
            raise ValueError(
                f"ACTION_DEPENDENCIES in {version}: unknown blocker '{blocker_name}' "
                f"(known actions: {sorted(name_to_idx)})"
            )
        blocker_idx = name_to_idx[blocker_name]
        resolved: set[int] = set()
        for blocked_name in blocked_names:
            if blocked_name not in name_to_idx:
                raise ValueError(
                    f"ACTION_DEPENDENCIES in {version}: unknown blocked '{blocked_name}' "
                    f"under blocker '{blocker_name}' (known actions: {sorted(name_to_idx)})"
                )
            blocked_idx = name_to_idx[blocked_name]
            if blocked_idx == blocker_idx:
                continue
            resolved.add(blocked_idx)
        if resolved:
            blocks[blocker_idx] = frozenset(resolved)

    raw_graph: dict[str, list[list[str]]] = getattr(mod, "SCHEDULE_GRAPH", {})
    schedule_paths: dict[str, tuple[tuple[int, ...], ...]] = {}
    for family, paths in raw_graph.items():
        resolved_paths: list[tuple[int, ...]] = []
        for path in paths:
            resolved_path: list[int] = []
            for action_name in path:
                if action_name not in name_to_idx:
                    raise ValueError(
                        f"SCHEDULE_GRAPH in {version}: unknown action '{action_name}' "
                        f"in family '{family}' (known actions: {sorted(name_to_idx)})"
                    )
                resolved_path.append(name_to_idx[action_name])
            resolved_paths.append(tuple(resolved_path))
        schedule_paths[family] = tuple(resolved_paths)

    return ActionRegistry(
        action_classes=classes,
        num_actions=n,
        done_idx=n,
        total_actions=n + 1,
        name_to_idx=name_to_idx,
        blocks=blocks,
        schedule_paths=schedule_paths,
    )
