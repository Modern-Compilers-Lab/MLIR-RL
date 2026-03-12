from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Generalization(ActionBase):
    """
    Convert a named structured operation (e.g., linalg.matmul,
    linalg.conv_2d_nchw_fchw) into its equivalent generic loop-nest form
    (linalg.generic), exposing all loop dimensions and indexing maps.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        # Must have a named linalg op (not already generic)
        # Check for common named ops
        has_named = any(
            op in code
            for op in [
                "linalg.matmul",
                "linalg.conv_2d_nchw_fchw",
                "linalg.conv_2d_nhwc_hwcf",
                "linalg.batch_matmul",
                "linalg.matvec",
                "linalg.vecmat_transpose",
                "linalg.dot",
            ]
        )
        return has_named

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %generic = transform.structured.generalize %op'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %generic "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        # After generalization, should contain linalg.generic
        if "linalg.generic" not in after:
            return False
        return True
