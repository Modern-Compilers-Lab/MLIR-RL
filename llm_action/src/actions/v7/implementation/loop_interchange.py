from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """
    Permutes the ordering of iterator dimensions in a tagged linalg operation
    to improve memory access patterns and spatial locality.

    The implementation first generalizes named linalg ops (matmul, conv2d, etc.)
    to linalg.generic, then applies the interchange. Already-generic ops pass
    through generalization unchanged.

    Uses transform.structured.generalize followed by
    transform.structured.interchange on the operation matched by
    tag = "operation_0". Re-annotates the result with the tag.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "Permutation of loop indices as a list of integers (0-based). "
                "Must be a valid permutation of [0, 1, ..., n-1].",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        perm = params.get("permutation", [])
        if not perm:
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
        perm = params["permutation"]
        perm_str = "[" + ", ".join(str(p) for p in perm) + "]"

        # Single transform sequence: generalize then interchange
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main("
            f"%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} '
            f"in %arg1 : (!transform.any_op) -> !transform.any_op\n"
            f"    %generic = transform.structured.generalize %op "
            f": (!transform.any_op) -> !transform.any_op\n"
            f"    %interchanged = transform.structured.interchange %generic "
            f"iterator_interchange = {perm_str} "
            f": (!transform.any_op) -> !transform.any_op\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f"    transform.annotate %interchanged "
            f'"tag" = %tag : !transform.any_op, !transform.any_param\n'
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
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
