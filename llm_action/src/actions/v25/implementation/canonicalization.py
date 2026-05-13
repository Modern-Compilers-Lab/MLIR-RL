from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Canonicalization(ActionBase):
    """
    Simplify and normalize IR by folding constants, eliminating dead code, and
    resolving dynamic shapes into static types. Zero-parameter cleanup action.
    """

    unique_execution: bool = True  # idempotent; second application is a no-op

    @classmethod
    def parameters(cls) -> dict:
        return {}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        # Generalize named ops to generic form, then canonicalize.
        # Generalize ensures observable IR change; canonicalize cleans up.
        transform_code = (
            'module attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n'
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            '    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n'
            '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            '    transform.annotate %generic "tag" = %tag : !transform.any_op, !transform.any_param\n'
            '    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            '    transform.apply_registered_pass "canonicalize" to %func : (!transform.any_op) -> !transform.any_op\n'
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

    @classmethod
    def params_size(cls) -> int:
        return 0

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return []

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {}
