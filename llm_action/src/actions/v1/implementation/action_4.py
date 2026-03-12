import re
from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class VectorizationAction(ActionBase):
    """
    Vectorization Action: Maps loop dimensions of a tagged linalg operation
    onto SIMD vector lanes using transform.structured.vectorize.

    The vector_sizes parameter specifies the vector dimensions. Sizes must
    be >= the corresponding iteration space dimensions of the target op.
    Typically applied after tiling to match tile sizes.

    Parameters:
        vector_sizes (list[int]): Vector sizes for each loop dimension.
            Must be >= iteration space sizes. All must be positive.
    """

    MAX_VECTOR_ELEMENTS = 1024
    MAX_VECTOR_RANK = 3
    # For very small vectors (total elements <= this), higher ranks are allowed
    SMALL_VECTOR_THRESHOLD = 64

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector sizes for each loop dimension. Must be >= iteration space sizes.",
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
        """Check that generated vectors comply with the vectorization safety contract."""
        vector_pattern = re.compile(r'vector<([^>]+)>')
        for match in vector_pattern.finditer(transformed_code):
            dims_str = match.group(1)
            # Extract numeric dimensions (ignore type like f64, f32)
            parts = dims_str.replace('x', ' ').split()
            dims = []
            for p in parts:
                try:
                    dims.append(int(p))
                except ValueError:
                    continue  # type string like "f64"

            if len(dims) == 0:
                continue

            # Check total element count
            total = 1
            for d in dims:
                total *= d
            if total > cls.MAX_VECTOR_ELEMENTS:
                return False

            # Check rank: allow higher ranks only for very small vectors
            if len(dims) > cls.MAX_VECTOR_RANK and total > cls.SMALL_VECTOR_THRESHOLD:
                return False

        return True

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]
        sizes_str = "[" + ", ".join(str(s) for s in vector_sizes) + "]"

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.structured.vectorize %op vector_sizes {sizes_str}'
            f' : !transform.any_op\n'
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
        if len(after.strip()) == 0:
            return False
        return True
