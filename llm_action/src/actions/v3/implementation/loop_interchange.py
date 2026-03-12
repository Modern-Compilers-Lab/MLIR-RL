from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """
    Permute the ordering of iterators in a linalg operation to place the loop
    with the most contiguous memory access pattern in the innermost position.
    Uses transform.structured.interchange.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "iterator_interchange": {
                "description": "Permutation of iterator indices. Must be a valid permutation of [0, 1, ..., n-1].",
                "type": "list[int]",
                "default": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        perm = params.get("iterator_interchange")
        if not perm or not isinstance(perm, list):
            return False
        if not all(isinstance(p, int) and p >= 0 for p in perm):
            return False
        if sorted(perm) != list(range(len(perm))):
            return False
        if perm == list(range(len(perm))):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        perm = params["iterator_interchange"]
        perm_str = ", ".join(str(p) for p in perm)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %op iterator_interchange = [{perm_str}] : (!transform.any_op) -> !transform.any_op\n'
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
