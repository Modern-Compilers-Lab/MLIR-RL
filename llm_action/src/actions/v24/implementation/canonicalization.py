from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Canonicalization(ActionBase):
    """
    Apply canonicalization pass to simplify and normalize IR by folding
    redundant operations, propagating constants, and cleaning up
    transformation artifacts.

    This action applies the registered 'canonicalize' pass to func.func
    operations in the module. It is most useful after other structural
    transforms (tiling, interchange, promotion) that introduce redundant
    operations.

    unique_execution = False because canonicalization can be applied after
    each structural transformation in a schedule.
    """

    unique_execution: bool = False  # useful to apply after each transform

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
