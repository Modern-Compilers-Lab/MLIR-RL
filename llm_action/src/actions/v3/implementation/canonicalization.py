from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Canonicalization(ActionBase):
    """
    Apply semantics-preserving simplifications to the IR, such as constant folding,
    dead code elimination, and operation normalization. Uses transform.apply_patterns
    with canonicalization patterns.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        if "func.func" not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        transform_code = (
            'module attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {\n'
            '    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            '    transform.apply_patterns to %func {\n'
            '      transform.apply_patterns.canonicalization\n'
            '    } : !transform.any_op\n'
            '    transform.yield\n'
            '  }\n'
            '}\n'
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
