from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code

import math


class Vectorization(ActionBase):
    """Tile to vector sizes and vectorize the linalg operation.

    Maps innermost loop iterations to SIMD vector operations using AVX2 vector
    lanes.  Includes tiling as a preprocessing step to match vector sizes.
    This is a lowering transform — the linalg op is consumed and replaced by
    vector + scf.for ops.
    """

    # Vectorization consumes the linalg op; a second application has no valid target.
    unique_execution: bool = True

    VOCAB = [2, 4, 8, 16]  # SIMD-friendly widths for f64

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector sizes per loop dimension for tiling and vectorization.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes", [])
        if not vector_sizes or not isinstance(vector_sizes, list):
            return False
        # All sizes must be positive integers from VOCAB
        if any(not isinstance(s, int) or s <= 0 for s in vector_sizes):
            return False
        # Check vector size product limit
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
        loop_results = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {vector_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {vector_sizes} : !transform.any_op\n'
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
        if 'tag = "operation_0"' not in after:
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
        # Enforce vector size product limit by clamping
        product = math.prod(sizes)
        while product > VECTORIZATION_SIZE_LIMIT and any(s > 2 for s in sizes):
            # Reduce the largest size
            max_idx = sizes.index(max(sizes))
            idx_in_vocab = cls.VOCAB.index(sizes[max_idx])
            if idx_in_vocab > 0:
                sizes[max_idx] = cls.VOCAB[idx_in_vocab - 1]
            product = math.prod(sizes)
        return {"vector_sizes": sizes}
