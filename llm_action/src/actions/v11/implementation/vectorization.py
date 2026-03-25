import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Vectorizes a tagged linalg operation using transform.structured.vectorize.

    Maps innermost loop iterations onto SIMD vector lanes, replacing scalar
    operations with vector instructions. Tiles the operation first to match
    the requested vector sizes.
    """

    VOCAB = [1, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector sizes per loop dimension. All must be positive. Product must be <= 1024.",
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
        if not all(isinstance(s, int) and s > 0 for s in vector_sizes):
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
    def _validate_vectors(cls, code: str) -> bool:
        """Check that all vector types in the output satisfy safety constraints."""
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

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]
        n_loops = len(vector_sizes)
        r = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {vector_sizes} : (!transform.any_op) -> (!transform.any_op, {r})\n"
            f"    transform.structured.vectorize %tiled_op vector_sizes {vector_sizes} : !transform.any_op\n"
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
        product = 1
        for s in sizes:
            product *= s
        while product > VECTORIZATION_SIZE_LIMIT and max(sizes) > 1:
            max_idx = sizes.index(max(sizes))
            sizes[max_idx] = max(1, sizes[max_idx] // 2)
            product = 1
            for s in sizes:
                product *= s
        return {"vector_sizes": sizes}
