from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Decomposition(ActionBase):
    """
    Break a compound operation into a sequence of simpler operations over sub-problems,
    exposing intermediate results and enabling per-stage optimization.
    Uses transform.structured.decompose to lower higher-dimensional operations
    into combinations of lower-dimensional equivalents.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        # Decomposition is applicable to higher-dimensional ops like convolutions
        decomposable_ops = [
            "linalg.conv_2d_nchw_fchw", "linalg.conv_2d_nhwc_hwcf",
            "linalg.depthwise_conv_2d_nhwc_hwc",
            "linalg.conv_1d", "linalg.batch_matmul",
        ]
        # Also works on linalg.generic that represents decomposable patterns
        has_decomposable = any(op in code for op in decomposable_ops)
        has_generic = "linalg.generic" in code
        return has_decomposable or has_generic

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %decomposed = transform.structured.decompose %op : (!transform.any_op) -> !transform.any_op\n'
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
