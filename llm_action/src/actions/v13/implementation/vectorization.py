import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Vectorize a linalg operation by tiling to vector sizes then applying vectorize."""

    VOCAB = [1, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector sizes per loop dimension for vectorization.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def _validate_vector_sizes(cls, transformed_code: str) -> bool:
        """Check that generated vectors respect size and rank constraints."""
        vector_pattern = re.findall(r"vector<([^>]+)>", transformed_code)
        for v in vector_pattern:
            dims_str = v.split("x")
            dims = []
            for d in dims_str:
                d = d.strip()
                if d and d[0].isdigit():
                    dims.append(int(d))
            if not dims:
                continue
            product = 1
            for d in dims:
                product *= d
            if product > VECTORIZATION_SIZE_LIMIT:
                return False
        return True

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes", [])
        if not vector_sizes or all(s == 0 for s in vector_sizes):
            return False
        if "linalg." not in code:
            return False
        product = 1
        for s in vector_sizes:
            if s > 0:
                product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        code = cls.preprocess(code, params)
        vector_sizes = params["vector_sizes"]

        n_loops = sum(1 for s in vector_sizes if s != 0)
        if n_loops == 0:
            return code

        # Tile to vector sizes first, then vectorize
        loop_types = ", ".join(["!transform.any_op"] * n_loops)
        sizes_str = str(vector_sizes)

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {sizes_str}"
            f" : (!transform.any_op) -> (!transform.any_op, {loop_types})\n"
            f"    transform.structured.vectorize %tiled_op vector_sizes {sizes_str} : !transform.any_op\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )

        try:
            result = run_transform_code(code, transform_code)
            if not cls._validate_vector_sizes(result):
                return code
            return result
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if before.strip() == after.strip():
            return False
        if "func.func" not in after:
            return False
        if "vector" not in after:
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
        # Clamp product to VECTORIZATION_SIZE_LIMIT
        product = 1
        for s in sizes:
            product *= max(s, 1)
        while product > VECTORIZATION_SIZE_LIMIT and any(s > 1 for s in sizes):
            for i in range(len(sizes) - 1, -1, -1):
                if sizes[i] > 1:
                    sizes[i] = max(1, sizes[i] // 2)
                    break
            product = 1
            for s in sizes:
                product *= max(s, 1)
        return {"vector_sizes": sizes}
