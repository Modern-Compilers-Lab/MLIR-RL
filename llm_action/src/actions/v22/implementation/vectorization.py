from math import prod

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """
    Replace scalar loop iterations with SIMD vector operations, mapping multiple
    data elements to a single vector instruction.

    Includes tiling as a preprocessing step to match vector sizes.
    This is a lowering transform that consumes the linalg op, so it is single-shot.
    """

    unique_execution: bool = True  # consumes linalg op, lowers to vector ops

    VOCAB = [2, 4, 8, 16]  # vector sizes for f64 (AVX2: 4 lanes)

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "SIMD vector width along each loop dimension.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes", [])
        if not vector_sizes:
            return False
        if all(s == 0 for s in vector_sizes):
            return False
        total = prod(s for s in vector_sizes if s > 0)
        if total > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]
        n_loops = len(vector_sizes)
        loop_handles = ", ".join(["!transform.any_op"] * n_loops)
        sizes_str = str(vector_sizes)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {sizes_str} : (!transform.any_op) -> (!transform.any_op, {loop_handles})\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {sizes_str} : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        # Clamp product to vectorization size limit
        while prod(sizes) > VECTORIZATION_SIZE_LIMIT:
            max_idx = sizes.index(max(sizes))
            sizes[max_idx] = max(2, sizes[max_idx] // 2)
        return {"vector_sizes": sizes}
