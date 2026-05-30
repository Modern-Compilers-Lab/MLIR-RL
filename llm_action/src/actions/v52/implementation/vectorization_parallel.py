import re

import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


def _get_n_loops(code: str) -> int:
    """Extract number of loop dimensions from iterator_types in the code."""
    match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
    if not match:
        return 0
    return len([t.strip() for t in match.group(1).split(',')])


class VectorizationParallel(ActionBase):
    """Tile innermost loops using tile_using_forall then vectorize.

    Category B (lowering): linalg op is consumed, replaced by vector ops inside scf.forall.
    Tags the scf.forall op after vectorization.
    Preprocessing uses tile_using_forall (parallel tiling distributing outer tiles across threads).
    """

    VOCAB = [1, 2, 4, 8]
    unique_execution = True  # Consumes the linalg op; one-shot lowering

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector sizes per loop dimension. Must divide iteration-space dims.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vsizes = params.get("vector_sizes", [])
        if not vsizes:
            return False
        if all(v == 1 for v in vsizes):
            return False  # all-ones is effectively a no-op vectorization
        if any(v <= 0 for v in vsizes):
            return False
        product = 1
        for v in vsizes:
            product *= v
        if product > 2048:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = list(params["vector_sizes"])
        n_loops = _get_n_loops(code)

        # Must tile ALL dims to avoid huge vectors on untiled dims
        if n_loops > 0:
            if len(vector_sizes) < n_loops:
                vector_sizes.extend([1] * (n_loops - len(vector_sizes)))
            elif len(vector_sizes) > n_loops:
                vector_sizes = vector_sizes[:n_loops]

        # All dims must have positive tile sizes
        for i in range(len(vector_sizes)):
            if vector_sizes[i] <= 0:
                vector_sizes[i] = 1

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op:2 = transform.structured.tile_using_forall %op tile_sizes {vector_sizes} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    transform.structured.vectorize %tiled_op#0 vector_sizes {vector_sizes} : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op#1 "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )
        try:
            result = run_transform_code(code, transform_code)
            # Validate vector sizes in output
            for m in re.finditer(r'vector<([^>]+)>', result):
                dims_str = m.group(1)
                dim_parts = dims_str.split('x')
                dims = []
                for p in dim_parts:
                    try:
                        dims.append(int(p))
                    except ValueError:
                        continue
                if dims:
                    product = 1
                    for d in dims:
                        product *= d
                    if product > 2048:
                        return code  # Reject: vector too large
            return result
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if 'func.func' not in after:
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
                bound > 0 and bound % s == 0
                for s in cls.VOCAB
            ], dtype=bool)
            if not slot_mask.any():
                slot_mask[0] = True  # 1 always divides
            masks.append(slot_mask)
        return np.concatenate(masks)
