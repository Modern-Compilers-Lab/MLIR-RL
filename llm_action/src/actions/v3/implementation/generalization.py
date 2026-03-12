from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Generalization(ActionBase):
    """
    Convert a named structured operation (e.g., matmul, conv) into its equivalent
    generic loop-nest form (linalg.generic), exposing the full iteration space for
    arbitrary restructuring. Uses transform.structured.generalize.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        # Generalization only makes sense for named linalg ops, not already-generic ones
        named_ops = [
            "linalg.matmul", "linalg.conv_2d_nchw_fchw", "linalg.conv_2d_nhwc_hwcf",
            "linalg.batch_matmul", "linalg.matvec", "linalg.vecmat",
            "linalg.dot", "linalg.fill", "linalg.conv_1d",
        ]
        has_named_op = any(op in code for op in named_ops)
        return has_named_op

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %gen = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n'
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
        if "linalg.generic" not in after:
            return False
        return True
