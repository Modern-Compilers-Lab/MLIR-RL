import numpy as np
import re
from functools import reduce
from operator import mul

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class VectorizationParallel(ActionBase):
    """Tile innermost loops to SIMD-compatible extents using parallel forall-loops,
    then vectorize the resulting fixed-extent inner loops.

    Uses tile_using_forall for preprocessing: outer tiles are distributed across
    threads while inner tiles are sized for SIMD.

    Lowering transform (Category B): the linalg op is consumed and replaced by
    vector + scf.forall ops. A second application has no valid linalg target.
    """

    # Lowering transform — linalg op is consumed, second application has no target
    unique_execution: bool = True

    VOCAB = [1, 4, 8, 16, 32, 64]  # 1 = scalar (no vectorization on that dim)

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "SIMD width per loop dimension; 1 means scalar. "
                               "Preprocessing uses tile_using_forall to distribute outer tiles across threads.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes")
        if not vector_sizes or not isinstance(vector_sizes, list):
            return False
        if all(s == 1 for s in vector_sizes):
            return False  # all-scalar = no-op
        if any(not isinstance(s, int) or s < 1 for s in vector_sizes):
            return False
        product = reduce(mul, vector_sizes, 1)
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]
        tile_sizes = list(vector_sizes)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall = transform.structured.tile_using_forall %op'
            f' tile_sizes {str(tile_sizes)}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {str(vector_sizes)} : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %forall "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
        except Exception:
            return code

        # Validate vector sizes in output
        vector_pattern = re.compile(r'vector<([^>]+)>')
        for m in vector_pattern.finditer(result):
            dims_str = m.group(1)
            parts = dims_str.split("x")
            numeric_parts = [int(p) for p in parts if p.isdigit()]
            if numeric_parts:
                product = reduce(mul, numeric_parts, 1)
                if product > VECTORIZATION_SIZE_LIMIT:
                    return code
                if len(numeric_parts) > 3:
                    return code

        return result

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
        product = reduce(mul, sizes, 1)
        while product > VECTORIZATION_SIZE_LIMIT and any(s > 1 for s in sizes):
            max_idx = max(range(len(sizes)), key=lambda i: sizes[i])
            idx_in_vocab = cls.VOCAB.index(sizes[max_idx])
            if idx_in_vocab > 0:
                sizes[max_idx] = cls.VOCAB[idx_in_vocab - 1]
            product = reduce(mul, sizes, 1)
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
