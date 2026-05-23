import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Tile all dimensions then vectorize the tiled op to SIMD vector operations.

    Terminal: consumes the linalg op, producing vector/scf.for ops (Category B).
    """

    unique_execution: bool = True  # consumes linalg op, replaces with vector ops

    VOCAB = [1, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": "Per-loop tile/vector sizes. All must be > 0. Product <= 2048. Values: [1, 2, 4, 8, 16]."
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or any(s <= 0 for s in tile_sizes):
            return False
        product = 1
        for s in tile_sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        if product <= 1:
            return False  # all-ones = no meaningful vectorization
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_loops = len(tile_sizes)
        loop_types = ", ".join(["!transform.any_op"] * n_loops)

        # Tile ALL dims (required: untiled dims keep original large sizes causing huge vectors)
        # Then vectorize the tiled op with matching vector sizes
        # Tag the outermost loop (Category B: linalg op is consumed)
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {str(tile_sizes)} : (!transform.any_op) -> (!transform.any_op, {loop_types})\n"
            f"    transform.structured.vectorize %tiled_op vector_sizes {str(tile_sizes)} : !transform.any_op\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f"    transform.annotate %loops#0 \"tag\" = %tag : !transform.any_op, !transform.any_param\n"
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
        # Clamp product to VECTORIZATION_SIZE_LIMIT
        product = 1
        for s in sizes:
            product *= s
        while product > VECTORIZATION_SIZE_LIMIT and any(s > 1 for s in sizes):
            max_idx = max(range(n), key=lambda i: sizes[i])
            sizes[max_idx] = max(1, sizes[max_idx] // 2)
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
            slot_mask = np.array(
                [bound > 0 and bound % s == 0 for s in cls.VOCAB],
                dtype=bool,
            )
            if not slot_mask.any():
                slot_mask[0] = True  # 1 always divides
            masks.append(slot_mask)
        return np.concatenate(masks)
