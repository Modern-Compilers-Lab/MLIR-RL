import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class ParallelVectorization(ActionBase):
    """Tile dims to vector sizes using parallel forall loops, then vectorize.

    Category B (lowering): the linalg op is consumed. The forall loop is tagged.
    Combines vectorization with thread-level parallelism in one action.
    """

    unique_execution: bool = True  # consumes the linalg op, not repeatable

    VOCAB = [1, 2, 4, 8, 16, 32]  # vector sizes; 1 = pass-through dim

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "SIMD tile/vector sizes for each loop dimension.",
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
        if any(not isinstance(s, int) or s < 1 for s in vector_sizes):
            return False
        if all(s == 1 for s in vector_sizes):
            return False
        product = 1
        for s in vector_sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]

        # tile_using_forall tiles parallel dims; vectorize the inner tiled op
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall = transform.structured.tile_using_forall %op'
            f' tile_sizes {str(vector_sizes)}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {str(vector_sizes)}'
            f' : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %forall "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
            # Post-transform vector size validation
            import re
            for m in re.finditer(r'vector<([^>]+)>', result):
                dims_str = m.group(1)
                dims_part = dims_str.split('x')
                product = 1
                for d in dims_part:
                    d = d.strip()
                    if d and d[0].isdigit():
                        product *= int(d)
                if product > VECTORIZATION_SIZE_LIMIT:
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
        product = 1
        for s in sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            for i in range(len(sizes) - 1, -1, -1):
                sizes[i] = 1
                product = 1
                for s in sizes:
                    product *= s
                if product <= VECTORIZATION_SIZE_LIMIT:
                    break
        return {"vector_sizes": sizes}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n = min(n_loops, MAX_PARAM_SLOTS)
        masks = []
        for i in range(n):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            slot_mask = np.array([
                s == 1 or (bound > 0 and bound % s == 0)
                for s in cls.VOCAB
            ], dtype=bool)
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        return np.concatenate(masks)
