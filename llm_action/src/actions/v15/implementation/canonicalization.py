from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Canonicalization(ActionBase):
    """
    Normalize the IR by generalizing named linalg operations to their
    canonical linalg.generic form, then applying pattern-based algebraic
    simplifications and optionally common subexpression elimination (CSE).
    """

    CSE_OPTIONS = [0, 1]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "apply_cse": {
                "description": "Whether to also apply CSE. 0=no, 1=yes.",
                "type": "int",
                "values": cls.CSE_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        apply_cse = params.get("apply_cse", -1)
        if apply_cse not in (0, 1):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        apply_cse = params["apply_cse"]

        cse_line = ""
        if apply_cse:
            cse_line = (
                f'    transform.apply_cse to %f0 : !transform.any_op\n'
            )

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %generic = transform.structured.generalize %op'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %generic "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    %f0 = transform.structured.match ops{{["func.func"]}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_patterns to %f0 {{\n'
            f'      transform.apply_patterns.canonicalization\n'
            f'    }} : !transform.any_op\n'
            + cse_line +
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
        if "func.func" not in after:
            return False
        if after.strip() == before.strip():
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.CSE_OPTIONS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {"apply_cse": cls.CSE_OPTIONS[raw_slots[0] % len(cls.CSE_OPTIONS)]}
