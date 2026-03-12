from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Decomposition(ActionBase):
    """
    Break a compound structured operation into a sequence of simpler
    operations, exposing intermediate results and per-stage optimization
    opportunities. Uses transform.structured.decompose on the tagged op.
    This works for operations that have a defined decomposition pattern
    (e.g., softmax, certain conv patterns). For convolutions, consider
    using Generalization followed by other transforms instead.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        # decompose works on ops with defined decomposition
        # It generally doesn't work on basic linalg.matmul or linalg.conv directly
        # It works on higher-level ops like softmax, winograd ops, etc.
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %decomposed = transform.structured.decompose %op'
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
