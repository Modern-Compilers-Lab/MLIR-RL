from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Image2Col(ActionBase):
    """Convert a conv2d operation into an img2col + matmul-like contraction.

    Restructures linalg.conv_2d_nchw_fchw (7 loops: N, F, OH, OW, C, KH, KW)
    into a linalg.generic contraction with 4 loops: [batch, M, N, K]
    where M=OH*OW (spatial), N=F (filters), K=C*KH*KW (channels*kernel).

    Single-shot: the conv2d op is consumed and replaced; a second application
    has no valid target."""

    unique_execution: bool = True  # consumes the conv2d op

    @classmethod
    def parameters(cls) -> dict:
        return {}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        if "linalg.conv_2d_nchw_fchw" not in code:
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
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            '    %img2col, %transformed = transform.structured.convert_conv2d_to_img2col %op : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            '    %contraction = transform.get_producer_of_operand %transformed[0] : (!transform.any_op) -> !transform.any_op\n'
            '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            '    transform.annotate %contraction "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        if "linalg.conv_2d_nchw_fchw" in after:
            return False
        if "linalg.generic" not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [1]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {}
