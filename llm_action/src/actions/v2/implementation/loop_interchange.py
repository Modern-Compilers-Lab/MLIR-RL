from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """
    Loop Interchange action: reorders loop iterators of a linalg operation to
    improve spatial locality or enable more effective tiling/vectorization.

    For named linalg ops (matmul, conv_2d_*), first generalizes to linalg.generic,
    then applies iterator interchange. For linalg.generic ops, applies interchange directly.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "iterator_interchange": {
                "description": "Permutation of iterator dimensions as a list of integers (zero-based). Must be a valid permutation of [0, ..., n-1].",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        interchange = params.get("iterator_interchange")
        if not interchange or not isinstance(interchange, list):
            return False
        if not all(isinstance(i, int) and i >= 0 for i in interchange):
            return False
        n = len(interchange)
        if sorted(interchange) != list(range(n)):
            return False
        if interchange == list(range(n)):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        interchange = params["iterator_interchange"]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %generic'
            f' iterator_interchange = {interchange}'
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
