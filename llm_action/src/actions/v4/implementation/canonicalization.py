from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Canonicalization(ActionBase):
    """
    Apply semantics-preserving simplifications to the IR, including
    constant folding, dead code elimination, and operation normalization.
    Uses transform.apply_patterns.canonicalization on all func.func ops.
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
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %f0 = transform.structured.match ops{{["func.func"]}} in %arg0'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_patterns to %f0 {{\n'
            f'      transform.apply_patterns.canonicalization\n'
            f'    }} : !transform.any_op\n'
            f'    transform.apply_patterns to %f0 {{\n'
            f'      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes\n'
            f'    }} : !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            return run_transform_code(code, transform_code)
        except Exception:
            # Fallback: just canonicalize without linalg patterns
            transform_code_simple = (
                f'module attributes {{transform.with_named_sequence}} {{\n'
                f'  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.consumed}}) {{\n'
                f'    %f0 = transform.structured.match ops{{["func.func"]}} in %arg0'
                f' : (!transform.any_op) -> !transform.any_op\n'
                f'    transform.apply_patterns to %f0 {{\n'
                f'      transform.apply_patterns.canonicalization\n'
                f'    }} : !transform.any_op\n'
                f'    transform.yield\n'
                f'  }}\n'
                f'}}\n'
            )
            try:
                return run_transform_code(code, transform_code_simple)
            except Exception:
                return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        # Canonicalization on already-canonical code may produce identical output
        # This is acceptable — the postcondition checks for non-empty valid IR
        if "func.func" not in after:
            return False
        if not after.strip():
            return False
        # Allow identity transforms for canonicalization (it's a cleanup action)
        return True
