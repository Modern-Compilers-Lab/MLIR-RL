import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """
    Tile to small register-level tiles then vectorize with matching vector sizes,
    replacing scalar arithmetic with AVX2 SIMD vector operations.
    This is a one-shot lowering (linalg op consumed, replaced by scf.for + vector ops),
    so unique_execution = True.
    """

    # unique_execution = True: vectorization is a lowering transform; the linalg op is
    # consumed and replaced by vector.* ops. A second application has no valid linalg target.
    unique_execution: bool = True

    VOCAB = [4, 8, 16, 32]  # vector/tile sizes; no 0 since all dims must be tiled

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": (
                    "Per-loop tile sizes used both for tiling and for vector sizes. "
                    "All entries must be > 0. Product must be <= VECTORIZATION_SIZE_LIMIT."
                ),
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes:
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
        n = len(tile_sizes)
        if n == 0 or any(s <= 0 for s in tile_sizes):
            return code

        loop_types = ', '.join(['!transform.any_op'] * n)
        sizes_str = str(tile_sizes)

        if n == 1:
            loops_lhs = "%tiled_op, %loop"
            outermost_loop = "%loop"
        else:
            loops_lhs = f"%tiled_op, %loops:{n}"
            outermost_loop = "%loops#0"

        transform_code = (
            f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    {loops_lhs} = transform.structured.tile_using_for %op tile_sizes {sizes_str}'
            f' : (!transform.any_op) -> (!transform.any_op, {loop_types})\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {sizes_str} : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate {outermost_loop} "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        # Vectorization should produce vector ops
        if "vector." not in after:
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
        product = 1
        sizes = []
        for i in range(n):
            v = cls.VOCAB[raw_slots[i] % len(cls.VOCAB)]
            # Clamp to stay within vector size limit
            if product * v > VECTORIZATION_SIZE_LIMIT:
                valid_vals = [x for x in cls.VOCAB if product * x <= VECTORIZATION_SIZE_LIMIT]
                v = valid_vals[-1] if valid_vals else cls.VOCAB[0]
            sizes.append(v)
            product *= v
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
                # If nothing divides, keep the smallest value
                slot_mask[0] = True
            masks.append(slot_mask)
        return np.concatenate(masks)
