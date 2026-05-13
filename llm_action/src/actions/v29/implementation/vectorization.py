import math

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Tile the loop nest to SIMD-friendly sizes and lower to vector operations.

    Category B (lowering): consumes the linalg op, replaces with vector + loop ops.
    """

    unique_execution: bool = True  # Consumes the linalg op; a second application has no valid linalg target

    VOCAB = [2, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector/tile sizes per loop dimension for vectorization.",
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
        if any(s <= 0 for s in vector_sizes):
            return False
        product = math.prod(vector_sizes)
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]
        n_loops = len(vector_sizes)
        r = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f" : (!transform.any_op) -> !transform.any_op\n"
            f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op"
            f" tile_sizes {str(vector_sizes)} : (!transform.any_op) -> (!transform.any_op, {r})\n"
            f"    transform.structured.vectorize %tiled_op vector_sizes {str(vector_sizes)}"
            f" : !transform.any_op\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        # Clamp: ensure total product <= VECTORIZATION_SIZE_LIMIT
        while math.prod(sizes) > VECTORIZATION_SIZE_LIMIT and max(sizes) > cls.VOCAB[0]:
            max_idx = sizes.index(max(sizes))
            vocab_idx = cls.VOCAB.index(sizes[max_idx])
            if vocab_idx > 0:
                sizes[max_idx] = cls.VOCAB[vocab_idx - 1]
            else:
                break
        return {"vector_sizes": sizes}
