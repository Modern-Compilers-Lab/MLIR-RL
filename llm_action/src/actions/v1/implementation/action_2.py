from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchangeAction(ActionBase):
    """
    Loop Interchange Action: Reorders the iterators of a tagged linalg.generic
    operation to change the iteration order, improving memory access patterns.

    Note: transform.structured.interchange only works on linalg.generic ops.
    Named linalg ops (matmul, conv) must first be generalized.

    Parameters:
        iterator_interchange (list[int]): A permutation of the iterator indices.
            Length must equal the number of iterators in the target operation.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "iterator_interchange": {
                "description": "Permutation of iterator indices for the target linalg.generic op.",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False

        interchange = params.get("iterator_interchange", None)
        if interchange is None or not isinstance(interchange, list):
            return False

        if len(interchange) == 0:
            return False

        # Must be non-negative integers
        if not all(isinstance(i, int) and i >= 0 for i in interchange):
            return False

        # Must be a valid permutation (0..n-1)
        if sorted(interchange) != list(range(len(interchange))):
            return False

        # Identity permutation is a no-op
        if interchange == list(range(len(interchange))):
            return False

        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        interchange = params["iterator_interchange"]
        interchange_str = "[" + ", ".join(str(i) for i in interchange) + "]"

        # First generalize the op (in case it's a named linalg op like matmul),
        # then apply interchange on the resulting generic.
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %generic = transform.structured.generalize %op'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %generic'
            f' iterator_interchange = {interchange_str}'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
            return result
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        if len(after.strip()) == 0:
            return False
        return True
