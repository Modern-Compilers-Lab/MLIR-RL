import re
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Vectorizes a tagged linalg operation using transform.structured.vectorize.

    Maps computation to SIMD vector instructions by first tiling the operation
    to match the requested vector sizes, then applying vectorization on the
    tiled op.  The vectorize transform has no output handle, so re-annotation
    is NOT possible; the postcondition simply checks the output contains
    vector ops.
    """

    # Realistic SIMD widths for f64 on AVX2
    VOCAB = [2, 4, 8, 16, 32]

    # ------------------------------------------------------------------
    # ActionBase interface
    # ------------------------------------------------------------------

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": (
                    "Vector sizes per loop dimension (one int per dimension). "
                    "All must be positive. Product must be <= "
                    f"{VECTORIZATION_SIZE_LIMIT}."
                ),
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        # Tag must exist
        if 'tag = "operation_0"' not in code:
            return False
        # vector_sizes must be a non-empty list of positive ints
        vector_sizes = params.get("vector_sizes")
        if not vector_sizes or not isinstance(vector_sizes, list):
            return False
        if not all(isinstance(s, int) and s > 0 for s in vector_sizes):
            return False
        # Product constraint
        product = 1
        for s in vector_sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        # Tiling is folded into implement(); nothing to do here.
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]
        n_loops = len(vector_sizes)
        loop_types = ", ".join(["!transform.any_op"] * n_loops)

        # Combined tile-then-vectorize transform
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main('
            f'%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes'
            f'{{tag = "operation_0"}} in %arg1 '
            f': (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loops:{n_loops} = "
            f"transform.structured.tile_using_for %op "
            f"tile_sizes {vector_sizes} "
            f": (!transform.any_op) -> (!transform.any_op, {loop_types})\n"
            f"    transform.structured.vectorize %tiled_op "
            f"vector_sizes {vector_sizes} : !transform.any_op\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )

        try:
            result = run_transform_code(code, transform_code)
            if not cls._validate_vectors(result):
                return code
            return result
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        # Code must have changed
        if after.strip() == before.strip():
            return False
        # Output must contain vector ops
        if "vector" not in after:
            return False
        return True

    # ------------------------------------------------------------------
    # RL parameter interface
    # ------------------------------------------------------------------

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

        # Clamp product to <= VECTORIZATION_SIZE_LIMIT by halving the
        # largest dimension until the constraint is satisfied.
        product = 1
        for s in sizes:
            product *= s
        while product > VECTORIZATION_SIZE_LIMIT and max(sizes) > 2:
            max_idx = sizes.index(max(sizes))
            sizes[max_idx] = max(2, sizes[max_idx] // 2)
            product = 1
            for s in sizes:
                product *= s

        return {"vector_sizes": sizes}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _validate_vectors(cls, code: str) -> bool:
        """Check that all vector types in the output satisfy safety constraints.

        Scans for ``vector<...>`` types and verifies that the product of
        dimensions does not exceed VECTORIZATION_SIZE_LIMIT and that there
        are at most 3 dimensions.
        """
        for match in re.finditer(r"vector<([^>]+)>", code):
            dims_str = match.group(1)
            parts = dims_str.replace("x", " ").split()
            dims = []
            for p in parts:
                try:
                    dims.append(int(p))
                except ValueError:
                    pass
            if not dims:
                continue
            product = 1
            for d in dims:
                product *= d
            if product > VECTORIZATION_SIZE_LIMIT:
                return False
            if len(dims) > 3:
                return False
        return True
