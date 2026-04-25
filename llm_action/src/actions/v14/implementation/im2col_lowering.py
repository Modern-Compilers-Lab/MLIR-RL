from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Im2colLowering(ActionBase):
    """
    Lower a `linalg.conv_2d_*` target op into an explicit im2col contraction
    using `transform.structured.convert_conv2d_to_img2col`. The conversion
    materializes an unfolded input view and rewrites the convolution as a
    matmul-shaped `linalg.generic`, after which the mature matmul recipe
    (tiling, packing, vectorization, parallelization) becomes applicable.

    Parameters:
      - enable: int in {0, 1}. A value of 1 actually runs the conversion;
        a value of 0 is a no-op signal used for RL action-space consistency.
    """

    ENABLE_VOCAB = [0, 1]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "enable": {
                "description": (
                    "If 1, run the conv-to-img2col conversion on the target "
                    "op. If 0, do not apply the transformation."
                ),
                "type": "int",
                "values": cls.ENABLE_VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        enable = params.get("enable")
        if not isinstance(enable, int) or enable not in (0, 1):
            return False
        if enable == 0:
            return False
        # Only applicable to linalg.conv_2d_* targets.
        if "linalg.conv_2d" not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        transform_code = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op
    %img2col, %transformed = transform.structured.convert_conv2d_to_img2col %op : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tag = transform.param.constant "operation_0" -> !transform.any_param
    transform.annotate %transformed "tag" = %tag : !transform.any_op, !transform.any_param
    transform.yield
  }
}
"""
        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if not after or "func.func" not in after:
            return False
        if after.strip() == before.strip():
            return False
        if 'tag = "operation_0"' not in after:
            return False
        # After img2col, the conv op is replaced by a linalg.generic
        # contraction so the original conv_2d handle disappears.
        return "linalg.conv_2d" not in after

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.ENABLE_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        enable = cls.ENABLE_VOCAB[raw_slots[0] % len(cls.ENABLE_VOCAB)]
        return {"enable": enable}
