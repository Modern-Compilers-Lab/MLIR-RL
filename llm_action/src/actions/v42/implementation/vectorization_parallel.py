import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class VectorizationParallel(ActionBase):
    """Lower to SIMD vector operations using parallel tiling (tile_using_forall)
    as preprocessing, which distributes outer tiles across threads.

    Uses tile_using_forall for parallel dims (0 for reduction dims) to
    distribute work, then tile_using_for for all dims to create inner tiles
    matching vector sizes, then vectorizes the innermost op.

    Category B (lowering): the linalg op is consumed.
    unique_execution is True because the linalg op no longer exists after
    vectorization.
    """

    unique_execution: bool = True  # consumes the linalg op, cannot vectorize twice

    VOCAB = [1, 2, 4, 8, 16, 32]  # vector sizes per dim; all must be > 0

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "SIMD width per loop dimension (all must be > 0)",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        sizes = params.get("vector_sizes", [])
        if not sizes or not isinstance(sizes, list):
            return False
        if any(not isinstance(s, int) or s <= 0 for s in sizes):
            return False
        product = 1
        for s in sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        sizes = params["vector_sizes"]
        n_dims = len(sizes)

        # Build the forall tile_sizes: use the vector sizes for ALL dims.
        # tile_using_forall distributes ALL specified dims. For matmul-like
        # ops, the inner op after forall has dimensions matching the vector
        # sizes, so vectorize directly.
        forall_sizes = list(sizes)

        # After tile_using_forall, the inner op has all dims = vector sizes.
        # Vectorize directly with the same vector sizes.

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall = transform.structured.tile_using_forall %op'
            f' tile_sizes {str(forall_sizes)}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {str(sizes)} : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %forall "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        product = 1
        for s in sizes:
            product *= s
        while product > VECTORIZATION_SIZE_LIMIT and len(sizes) > 0:
            max_idx = sizes.index(max(sizes))
            idx_in_vocab = cls.VOCAB.index(sizes[max_idx])
            if idx_in_vocab > 0:
                sizes[max_idx] = cls.VOCAB[idx_in_vocab - 1]
            else:
                break
            product = 1
            for s in sizes:
                product *= s
        return {"vector_sizes": sizes}

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
                slot_mask[0] = True
            masks.append(slot_mask)
        return np.concatenate(masks)
