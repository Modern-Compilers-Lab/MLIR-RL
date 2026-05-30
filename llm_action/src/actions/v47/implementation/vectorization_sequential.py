import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class VectorizationSequential(ActionBase):
    """Tile inner loops to SIMD-width granularity using sequential for-loops,
    then lower the resulting fixed-trip-count loops to vector operations.

    Preprocessing tiles the loop nest using tile_using_for to create
    vector-width inner trips, then vectorizes the tiled op. The outermost
    generated loop is tagged for subsequent actions.

    This is a Category B (lowering) transform: the linalg op is consumed
    and replaced by vector + scf ops.

    unique_execution = True: vectorization consumes the linalg op;
    a second application has no valid linalg target.
    """

    unique_execution: bool = True  # lowering: consumes the linalg op

    VOCAB = [1, 4, 8, 16, 32, 64]  # 1 = scalar (no vectorization) for that dimension

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "SIMD width per loop dimension; 1 means scalar.",
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
        if all(s <= 1 for s in vector_sizes):
            return False  # all-scalar = no-op
        if any(not isinstance(s, int) or s < 1 for s in vector_sizes):
            return False
        # Check vector size product within safety limit
        product = 1
        for s in vector_sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        # Check rank constraint
        if len(vector_sizes) > 3:
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
        sizes_str = str(vector_sizes)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op'
            f' tile_sizes {sizes_str} : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {sizes_str} : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
            # Post-transform vector safety check
            if not cls._check_vector_safety(result):
                return code
            return result
        except Exception:
            return code

    @classmethod
    def _check_vector_safety(cls, code: str) -> bool:
        """Reject transforms that produce vectors exceeding safety limits."""
        import re
        for m in re.finditer(r'vector<([^>]+)>', code):
            dims_str = m.group(1)
            # Extract numeric dimensions (ignore type like f64)
            parts = dims_str.replace("x", " ").split()
            product = 1
            for p in parts:
                try:
                    product *= int(p)
                except ValueError:
                    pass
            if product > VECTORIZATION_SIZE_LIMIT:
                return False
        return True

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        return True

    # ---- RL interface ----

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
        # Clamp product to within vectorization size limit
        product = 1
        for s in sizes:
            product *= s
        while product > VECTORIZATION_SIZE_LIMIT and sizes:
            max_idx = max(range(len(sizes)), key=lambda i: sizes[i])
            if sizes[max_idx] <= 1:
                break
            sizes[max_idx] = max(1, sizes[max_idx] // 2)
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
                slot_mask[0] = True  # 1 is always safe (scalar)
            masks.append(slot_mask)
        return np.concatenate(masks)
