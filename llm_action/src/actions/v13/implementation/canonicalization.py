from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Canonicalization(ActionBase):
    """Apply canonicalization and CSE patterns to simplify IR."""

    @classmethod
    def parameters(cls) -> dict:
        return {}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if "func.func" not in code:
            return False
        # Only useful after structural transforms have been applied
        has_complexity = any(
            pattern in code
            for pattern in [
                "scf.for",
                "scf.forall",
                "tensor.extract_slice",
                "tensor.insert_slice",
                "tensor.pad",
                "affine.apply",
                "affine.min",
                "arith.constant",
                "linalg.generic",
            ]
        )
        if not has_complexity:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        code = cls.preprocess(code, params)

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %func = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    transform.apply_patterns to %func {{\n"
            f"      transform.apply_patterns.canonicalization\n"
            f"    }} : !transform.any_op\n"
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
        if before.strip() == after.strip():
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
