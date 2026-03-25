import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code
from llm_action.src.config import VECTORIZATION_SIZE_LIMIT


class Vectorization(ActionBase):
    """
    Vectorizes a tagged linalg operation by first tiling it to the requested
    vector sizes and then applying vectorization in a single transform sequence.

    Uses transform.structured.tile_using_for followed by
    transform.structured.vectorize on the operation matched by
    tag = "operation_0".

    Note: After vectorization the linalg operation is replaced by vector ops,
    so the tag "operation_0" is consumed and cannot be re-annotated.
    Vectorization is typically a terminal transformation for a structured op.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector sizes for each loop dimension of the operation. "
                "All entries must be positive integers. The product of sizes must "
                "not exceed 1024 to ensure hardware-realistic SIMD vectors.",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        sizes = params.get("vector_sizes", [])
        if not sizes or any(s <= 0 for s in sizes):
            return False
        total = 1
        for s in sizes:
            total *= s
        if total > VECTORIZATION_SIZE_LIMIT:
            return False
        rank = len(sizes)
        if rank >= 3 and total > 64:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        sizes = params["vector_sizes"]
        n_loops = len(sizes)
        loop_results = ", ".join(["!transform.any_op"] * n_loops)
        sizes_str = ", ".join(str(s) for s in sizes)

        # Single transform sequence: tile then vectorize
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main("
            f"%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} '
            f"in %arg1 : (!transform.any_op) -> !transform.any_op\n"
            f"    %tiled_op, %loops:{n_loops} = "
            f"transform.structured.tile_using_for %op tile_sizes [{sizes_str}] "
            f": (!transform.any_op) -> (!transform.any_op, {loop_results})\n"
            f"    transform.structured.vectorize %tiled_op "
            f"vector_sizes [{sizes_str}] : !transform.any_op\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )

        try:
            result = run_transform_code(code, transform_code)
        except Exception:
            return code

        # Validate vector sizes in output per the vectorization safety contract
        vectors = re.findall(r"vector<([^>]+)>", result)
        for vec_spec in vectors:
            dims = vec_spec.split("x")
            numeric_dims = []
            for d in dims:
                try:
                    numeric_dims.append(int(d))
                except ValueError:
                    continue
            if numeric_dims:
                n = 1
                for d in numeric_dims:
                    n *= d
                if n > VECTORIZATION_SIZE_LIMIT:
                    return code
                if len(numeric_dims) >= 3 and n > 64:
                    return code

        return result

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        return True
