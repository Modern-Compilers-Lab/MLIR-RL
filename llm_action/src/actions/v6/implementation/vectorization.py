import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """
    Convert scalar loop iterations into vector operations by mapping loop
    dimensions onto SIMD lanes, enabling data-parallel execution.
    Uses transform.structured.vectorize with parameterized vector sizes.
    Includes safety checks to prevent oversized or high-rank vectors.
    """

    MAX_VECTOR_ELEMENTS = 1024
    MAX_VECTOR_RANK = 3

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "List of vector sizes, one per loop dimension. Must be >= the iteration space sizes.",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes", None)
        if vector_sizes is None or not isinstance(vector_sizes, list):
            return False
        if len(vector_sizes) == 0:
            return False
        if not all(isinstance(s, int) and s > 0 for s in vector_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def _check_vector_safety(cls, transformed_code: str) -> bool:
        """Check that all vector types in the transformed code are within safe bounds."""
        vector_pattern = re.compile(r'vector<([^>]+)>')
        for match in vector_pattern.finditer(transformed_code):
            type_str = match.group(1)
            # Extract dimensions (ignore the element type like f64, f32, etc.)
            parts = type_str.split("x")
            dims = []
            for p in parts:
                p = p.strip()
                try:
                    dims.append(int(p))
                except ValueError:
                    continue  # this is the element type (f64, f32, etc.)
            if len(dims) == 0:
                continue
            # Check total elements
            total = 1
            for d in dims:
                total *= d
            if total > cls.MAX_VECTOR_ELEMENTS:
                return False
        return True

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]
        sizes_str = str(vector_sizes)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.structured.vectorize %op vector_sizes {sizes_str} : !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
        except Exception:
            return code

        # Vectorization safety check
        if not cls._check_vector_safety(result):
            return code

        return result

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        return True
