import re
from functools import reduce
from operator import mul

from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """
    Convert the innermost scalar loop into vector operations that process
    multiple elements simultaneously using SIMD instructions.
    Tiles the operation first, then vectorizes the tiled inner operation
    with vector_sizes matching the tile dimensions.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector sizes per loop dimension for the vectorized tile. "
                               "Must respect hardware SIMD width constraints.",
                "type": "list[int]",
                "default": None,
            },
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
        # Vectorization safety: bound total vector size
        total = reduce(mul, vector_sizes, 1)
        if total > 1024:
            return False
        # Disallow rank >= 3 vectors unless very small
        non_one = [s for s in vector_sizes if s > 1]
        if len(non_one) >= 3 and total > 256:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]

        n_loops = len(vector_sizes)
        loop_types = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op'
            f' tile_sizes {vector_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_types})\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {vector_sizes}'
            f' : !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
        except Exception:
            return code

        # Post-transform vector safety check
        vector_pattern = re.compile(r'vector<([^>]+)>')
        for match in vector_pattern.finditer(result):
            dims_str = match.group(1)
            # Extract numeric dimensions (ignore type like f64)
            parts = dims_str.replace('x', ' ').split()
            dims = []
            for p in parts:
                try:
                    dims.append(int(p))
                except ValueError:
                    pass
            if dims:
                total = reduce(mul, dims, 1)
                if total > 1024:
                    return code
                if len(dims) >= 3 and total > 256:
                    return code

        return result

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        return True
