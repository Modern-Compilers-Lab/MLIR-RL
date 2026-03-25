"""Build structured MultiDiscrete action spaces from action registry.

Each action declares its parameter slots via params_size() and classes_per_slot().
This module concatenates them into a single MultiDiscrete space compatible with
SB3's MaskablePPO, and provides a slot_map for unpacking.
"""

import numpy as np
from gymnasium import spaces

from llm_action.src.config import L


def build_action_space(registry, max_n_loops: int = L):
    """Build a MultiDiscrete action space from the registry.

    Layout:
      dim 0: action selector (registry.total_actions categories, including "done")
      dims 1..K: concatenated per-action parameter slots

    Each action's slots are sized by classes_per_slot(max_n_loops).
    Actions with params_size() == 0 get no parameter dimensions.

    Returns:
        (space, slot_map) where slot_map[action_idx] = (start, end) offsets
        into dims[1:]. The "done" action (registry.done_idx) maps to an empty
        range.
    """
    dims = [registry.total_actions]
    slot_map = {}
    offset = 0

    for idx, cls in enumerate(registry.action_classes):
        cps = cls.classes_per_slot(max_n_loops)
        n_slots = len(cps)
        slot_map[idx] = (offset, offset + n_slots)
        for c in cps:
            dims.append(max(c, 1))
        offset += n_slots

    # "done" pseudo-action has no parameters
    slot_map[registry.done_idx] = (offset, offset)

    return spaces.MultiDiscrete(np.array(dims, dtype=np.int64)), slot_map


def build_action_masks(registry, slot_map, n_loops: int, max_n_loops: int,
                       used_action_indices: set, unique_actions: bool) -> np.ndarray:
    """Build per-dimension boolean masks for MaskablePPO.

    Returns a flat bool array: [action_mask | param_slot_masks...].
    """
    # Action selector mask
    action_mask = np.ones(registry.total_actions, dtype=bool)
    if unique_actions:
        for idx in used_action_indices:
            action_mask[idx] = False
    action_mask[registry.done_idx] = True

    # Parameter slot masks
    param_masks = []
    for idx, cls in enumerate(registry.action_classes):
        start, end = slot_map[idx]
        n_slots = end - start

        max_cps = cls.classes_per_slot(max_n_loops)
        actual_cps = cls.classes_per_slot(n_loops)
        for slot_i in range(n_slots):
            slot_mask = np.zeros(max(max_cps[slot_i], 1), dtype=bool)
            actual = actual_cps[slot_i] if slot_i < len(actual_cps) else 0
            slot_mask[:max(actual, 1)] = True
            param_masks.append(slot_mask)

    if param_masks:
        return np.concatenate([action_mask] + param_masks)
    return action_mask
