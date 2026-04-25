from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Im2colLowering(ActionBase):
    """
    Convert a convolution operation into an image-to-column rearrangement
    followed by a matrix multiplication contraction, transforming the
    irregular sliding-window pattern into a regular dense operation.
    """

    ENABLE_OPTIONS = [0, 1]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "enable": {
                "description": "Whether to apply im2col conversion. 0=no, 1=yes.",
                "type": "int",
                "values": cls.ENABLE_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        enable = params.get("enable", 0)
        if enable != 1:
            return False
        if "conv_2d" not in code and "conv2d" not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %img2col, %transformed = transform.structured.convert_conv2d_to_img2col %op'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %matmul = transform.get_producer_of_operand %transformed[0]'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %matmul "tag" = %tag : !transform.any_op, !transform.any_param\n'
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

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.ENABLE_OPTIONS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {"enable": cls.ENABLE_OPTIONS[raw_slots[0] % len(cls.ENABLE_OPTIONS)]}
