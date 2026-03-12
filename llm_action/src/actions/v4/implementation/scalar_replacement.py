from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class ScalarReplacement(ActionBase):
    """
    Replace repeated memory loads of the same value within a loop body
    with a single load into a register, eliminating redundant memory accesses.
    This uses LICM (loop-invariant code motion) and canonicalization patterns
    to hoist invariant loads out of loops.
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
        # Need loops for LICM to be useful
        has_loops = "scf.for" in code or "scf.forall" in code or "affine.for" in code
        return has_loops

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %all_loops = transform.structured.match interface{{LoopLikeInterface}} in %arg0'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_licm to %all_loops : !transform.any_op\n'
            f'    %f0 = transform.structured.match ops{{["func.func"]}} in %arg0'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_patterns to %f0 {{\n'
            f'      transform.apply_patterns.canonicalization\n'
            f'    }} : !transform.any_op\n'
            f'    transform.apply_cse to %f0 : !transform.any_op\n'
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
        if not after.strip():
            return False
        # LICM/CSE may or may not change the code depending on the input
        return True
