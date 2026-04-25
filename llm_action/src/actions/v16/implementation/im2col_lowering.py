from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Im2colLowering(ActionBase):
    """
    Im2col Lowering action: transforms a convolution operation into a matrix
    multiplication by reorganizing input data using
    transform.structured.convert_conv2d_to_img2col.
    """

    ENABLE_VOCAB = [0, 1]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "enable": {
                "description": "Whether to apply im2col lowering (0=no, 1=yes)",
                "type": "int",
                "values": cls.ENABLE_VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        enable = params.get("enable", 0)
        if enable != 1:
            return False
        # Must contain a conv2d operation
        if "conv_2d" not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        transform_code = (
            '\nmodule attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n'
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1'
            ' : (!transform.any_op) -> !transform.any_op\n'
            '    %img2col, %transformed = transform.structured.convert_conv2d_to_img2col %op'
            ' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            '    %matmul = transform.get_producer_of_operand %transformed[0]'
            ' : (!transform.any_op) -> !transform.any_op\n'
            '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            '    transform.annotate %matmul "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.ENABLE_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {"enable": cls.ENABLE_VOCAB[raw_slots[0] % len(cls.ENABLE_VOCAB)]}
