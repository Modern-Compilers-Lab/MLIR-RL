import re
import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


# Known iterator types per op family
_ITERATOR_TYPES = {
    "linalg.matmul": ["parallel", "parallel", "reduction"],
    "linalg.conv_2d_nchw_fchw": ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"],
    "linalg.pooling_nchw_max": ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"],
    "linalg.add": ["parallel", "parallel", "parallel", "parallel"],
}


def _parse_iterator_types(code: str) -> list[str] | None:
    """Parse iterator_types from linalg.generic in code."""
    m = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
    if m:
        types = re.findall(r'"(\w+)"', m.group(1))
        return types
    for op_name, types in _ITERATOR_TYPES.items():
        if op_name in code:
            return types
    return None


class ParallelizationTiling(ActionBase):
    """Tile outer parallel dimensions and distribute tiles across threads.

    Uses tile_using_forall with tile_sizes. Reduction dimensions are automatically
    zeroed out to maintain correctness (forall cannot tile reduction dims).
    """

    unique_execution = True  # Introduces scf.forall, changes loop structure fundamentally

    VOCAB = [0, 4, 8, 16, 32, 64]  # 0 = no-tile; powers of 2

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for parallel distribution. 0 = do not tile. Reduction dims forced to 0.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or not isinstance(tile_sizes, list):
            return False
        if "linalg." not in code:
            return False
        iter_types = _parse_iterator_types(code)
        if iter_types is None:
            return False
        # Zero out reduction dims and check if anything remains
        effective = []
        for i, s in enumerate(tile_sizes):
            if i < len(iter_types) and iter_types[i] == "reduction":
                effective.append(0)
            else:
                effective.append(s)
        if all(s == 0 for s in effective):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        iter_types = _parse_iterator_types(code)
        if iter_types is None:
            return code

        # Zero out reduction dims
        effective = []
        for i, s in enumerate(tile_sizes):
            if i < len(iter_types) and iter_types[i] == "reduction":
                effective.append(0)
            else:
                effective.append(s)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled:2 = transform.structured.tile_using_forall %op tile_sizes {effective} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
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
        n = min(n_loops, MAX_PARAM_SLOTS)
        return [len(cls.VOCAB)] * n

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
            slot_mask = np.array([
                s == 0 or (bound > 0 and bound % s == 0)
                for s in cls.VOCAB
            ], dtype=bool)
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        return np.concatenate(masks)
