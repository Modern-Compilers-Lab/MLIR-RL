from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """
    Permute the ordering of iterator dimensions in a linalg operation to change
    the memory access pattern, improving spatial locality or enabling more
    efficient vectorization.
    Requires generalization to linalg.generic first (for named ops like matmul/conv).
    Uses transform.structured.generalize + transform.structured.interchange.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "A permutation of iterator indices specifying the new loop ordering.",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        permutation = params.get("permutation", None)
        if permutation is None or not isinstance(permutation, list):
            return False
        if len(permutation) == 0:
            return False
        if not all(isinstance(p, int) and p >= 0 for p in permutation):
            return False
        if sorted(permutation) != list(range(len(permutation))):
            return False
        if permutation == list(range(len(permutation))):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        permutation = params["permutation"]
        perm_str = str(permutation).replace("[", "[").replace("]", "]")

        # We always generalize first (converts named ops to linalg.generic),
        # then apply interchange. If the op is already generic, generalize is a no-op.
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %generic'
            f' iterator_interchange = {perm_str} : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %interchanged "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return True
