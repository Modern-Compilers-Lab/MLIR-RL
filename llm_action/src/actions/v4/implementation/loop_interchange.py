from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """
    Permute the ordering of loops within a loop nest to place the dimension
    with the most contiguous memory access in the innermost position.
    Requires the target operation to be a linalg.generic (use Generalization
    first on named ops). Uses transform.structured.interchange.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "Permutation vector for the iterator dimensions. "
                               "E.g. [1, 2, 0] reorders a 3-dim loop nest.",
                "type": "list[int]",
                "default": None,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        permutation = params.get("permutation")
        if not permutation or not isinstance(permutation, list):
            return False
        if not all(isinstance(p, int) and p >= 0 for p in permutation):
            return False
        # Must be a valid permutation
        if sorted(permutation) != list(range(len(permutation))):
            return False
        # Identity permutation is a no-op
        if permutation == list(range(len(permutation))):
            return False
        # Interchange requires linalg.generic (not named ops like matmul)
        if "linalg.generic" not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        permutation = params["permutation"]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %op'
            f' iterator_interchange = {permutation}'
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
