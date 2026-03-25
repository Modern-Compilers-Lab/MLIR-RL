import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code

MAX_VECTOR_ELEMENTS = 1024
MAX_VECTOR_RANK = 3


class Vectorization(ActionBase):
    """
    Replace scalar operations in a tagged linalg operation with SIMD
    vector operations. First tiles the operation to the requested
    vector_sizes, then vectorizes the tiled operation.

    Note: vectorize consumes the linalg op handle (no result handle),
    so tag re-annotation targets the tiled operation before vectorization.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "List of vector sizes, one per iteration dimension of the target operation.",
                "type": "list[int]",
            }
        }

    # Conv2d ops with windowed access patterns cannot be directly vectorized
    _NON_VECTORIZABLE_OPS = ["linalg.conv_2d", "linalg.conv_3d", "linalg.depthwise_conv"]

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        # Reject known non-vectorizable op patterns (windowed access)
        for op_name in cls._NON_VECTORIZABLE_OPS:
            if op_name in code:
                return False
        vector_sizes = params.get("vector_sizes", [])
        if not isinstance(vector_sizes, list) or len(vector_sizes) == 0:
            return False
        if not all(isinstance(s, int) and s > 0 for s in vector_sizes):
            return False
        # Check vectorization safety: total elements <= MAX_VECTOR_ELEMENTS
        total = 1
        for s in vector_sizes:
            total *= s
        if total > MAX_VECTOR_ELEMENTS:
            return False
        # Check rank constraint
        if len(vector_sizes) > MAX_VECTOR_RANK:
            # Allow higher rank only if very small
            if total > 64:
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
        results_type = f"(!transform.any_op, {loop_results})"

        if n_loops > 1:
            binding = f"%tiled_op, %loops:{n_loops}"
        else:
            binding = "%tiled_op, %loop"

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg0 : (!transform.any_op) -> !transform.any_op\n'
            f'    {binding} = transform.structured.tile_using_for %op tile_sizes {vector_sizes} : (!transform.any_op) -> {results_type}\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {vector_sizes} : !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
        except Exception:
            return code

        # Post-transform validation: check vector sizes in output
        vector_pattern = re.compile(r'vector<([^>]+)>')
        for match in vector_pattern.finditer(result):
            dims_str = match.group(1)
            parts = dims_str.replace('x', ' ').split()
            dims = []
            for p in parts:
                try:
                    dims.append(int(p))
                except ValueError:
                    pass
            if len(dims) > 0:
                total = 1
                for d in dims:
                    total *= d
                if total > MAX_VECTOR_ELEMENTS:
                    return code
                if len(dims) >= 3 and total > 64:
                    return code

        return result

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        return True
