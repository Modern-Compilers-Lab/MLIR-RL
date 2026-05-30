import re

import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


def _parse_iterator_types(code: str) -> list[str]:
    """Parse iterator types from linalg operation in MLIR code."""
    if "linalg.matmul" in code and "linalg.generic" not in code:
        return ["parallel", "parallel", "reduction"]
    m = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
    if m:
        return [t.strip().strip('"') for t in m.group(1).split(",")]
    return []


class ParallelizationTiling(ActionBase):
    """Distribute parallel loop dimensions via tiling across threads."""

    unique_execution = True  # introduces scf.forall, fundamentally changes loop structure

    VOCAB = [0, 4, 8, 16, 32, 64]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile size per loop dim; 0 skips; reduction dims auto-zeroed",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or all(s == 0 for s in tile_sizes):
            return False
        iter_types = _parse_iterator_types(code)
        if not iter_types:
            return False
        has_parallel = any(
            i < len(tile_sizes) and tile_sizes[i] != 0 and i < len(iter_types) and iter_types[i] == "parallel"
            for i in range(min(len(tile_sizes), len(iter_types)))
        )
        return has_parallel

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        iter_types = _parse_iterator_types(code)
        if not iter_types:
            return code

        n = min(len(tile_sizes), len(iter_types))
        # Zero out reduction dims for forall
        forall_tiles = [
            tile_sizes[i] if i < n and iter_types[i] == "parallel" else 0
            for i in range(n)
        ]

        if all(s == 0 for s in forall_tiles):
            return code

        sizes_str = str(forall_tiles).replace(" ", "")

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %forall = transform.structured.tile_using_forall %op tile_sizes {sizes_str}"
            f" : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )

        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"tile_sizes": sizes}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n = min(n_loops, MAX_PARAM_SLOTS)
        masks = []
        for i in range(n):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            slot_mask = np.array(
                [s == 0 or (bound > 0 and bound % s == 0) for s in cls.VOCAB],
                dtype=bool,
            )
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        return np.concatenate(masks)
