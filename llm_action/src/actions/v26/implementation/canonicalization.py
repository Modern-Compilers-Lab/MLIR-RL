from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Canonicalization(ActionBase):
    """Apply MLIR canonicalization and CSE passes to simplify IR.
    Useful between transformation steps to normalize intermediate IR."""

    unique_execution = False  # Canonicalization is useful between different transform steps

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
            '  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n'
            '    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            '    transform.apply_registered_pass "canonicalize" to %func : (!transform.any_op) -> !transform.any_op\n'
            '    %func2 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            '    transform.apply_registered_pass "cse" to %func2 : (!transform.any_op) -> !transform.any_op\n'
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
        if "func.func" not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return 0

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return []

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {}
