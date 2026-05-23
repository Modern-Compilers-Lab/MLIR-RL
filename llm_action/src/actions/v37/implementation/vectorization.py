import re
import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Tile-then-vectorize: tile all dims to vector-friendly sizes, then vectorize.

    This is a lowering transform (Category B) — the linalg op is consumed and
    replaced by vector ops inside scf.for loops. The outermost loop is tagged.
    """

    unique_execution: bool = True  # linalg op is consumed; cannot vectorize twice

    VOCAB = [1, 2, 4, 8, 16]  # all must be > 0 for vectorization

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Per-loop tile/vector sizes. All must be > 0. Product must be <= 2048.",
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
        if any(s <= 0 for s in tile_sizes):
            return False
        product = 1
        for s in tile_sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_loops = len(tile_sizes)  # all sizes > 0, so all produce loops

        loop_results = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {tile_sizes} : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
            # Validate vector sizes in the output
            for m in re.finditer(r'vector<([^>]+)>', result):
                dims_str = m.group(1)
                # Parse e.g. "1x1x5x5xf64"
                parts = dims_str.replace('x', ' ').split()
                numeric = [int(p) for p in parts if p.isdigit()]
                if numeric:
                    product = 1
                    for d in numeric:
                        product *= d
                    if product > VECTORIZATION_SIZE_LIMIT:
                        return code
                    if len(numeric) >= 3 and product > 64:
                        return code
            return result
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
        # Clamp product to VECTORIZATION_SIZE_LIMIT
        product = 1
        for s in sizes:
            product *= s
        while product > VECTORIZATION_SIZE_LIMIT and len(sizes) > 0:
            for i in range(len(sizes) - 1, -1, -1):
                if sizes[i] > 1:
                    idx = cls.VOCAB.index(sizes[i])
                    sizes[i] = cls.VOCAB[max(0, idx - 1)]
                    break
            product = 1
            for s in sizes:
                product *= s
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
                bound > 0 and bound % s == 0
                for s in cls.VOCAB
            ], dtype=bool)
            if not slot_mask.any():
                slot_mask[0] = True  # size 1 always divides
            masks.append(slot_mask)
        return np.concatenate(masks)
