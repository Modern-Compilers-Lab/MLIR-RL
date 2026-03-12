from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Packing(ActionBase):
    """
    Packing action: applies data tiling (packing) to a linalg operation,
    copying tiles of data into contiguous temporary buffers with optimized
    layout for the subsequent computation. Uses transform.structured.pack.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "description": "List of pack sizes, one per iterator dimension. 0 means do not pack that dimension.",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        packed_sizes = params.get("packed_sizes")
        if not packed_sizes or not isinstance(packed_sizes, list):
            return False
        if not all(isinstance(s, int) and s >= 0 for s in packed_sizes):
            return False
        if all(s == 0 for s in packed_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        packed_sizes = params["packed_sizes"]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %packed = transform.structured.pack %op'
            f' packed_sizes = {packed_sizes}'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        return True
