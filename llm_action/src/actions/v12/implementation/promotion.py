from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """Promotes operand slices into contiguous local buffers via transform.structured.pad.

    Since our RL system works with tensor semantics (not memref),
    transform.structured.promote is not applicable.  Instead we use
    transform.structured.pad, which pads operand tensors to multiples of
    a specified size, achieving the same effect as promotion in tensor
    land -- it creates clean, aligned copies of the data.
    """

    PAD_OPTIONS = [2, 4, 8, 16, 32]

    # ------------------------------------------------------------------
    # ActionBase interface
    # ------------------------------------------------------------------

    @classmethod
    def parameters(cls) -> dict:
        return {
            "pad_multiple": {
                "description": "Pad operand dimensions to multiples of this value.",
                "type": "int",
                "values": cls.PAD_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        pad_multiple = params.get("pad_multiple")
        if not isinstance(pad_multiple, int) or pad_multiple <= 1:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        pad_multiple = params["pad_multiple"]
        m = pad_multiple

        # 3 padding values / dimensions for matmul (input A, input B, output C)
        pad_to_multiple_of = f"[{m}, {m}, {m}]"
        padding_values = "[0.0 : f64, 0.0 : f64, 0.0 : f64]"
        padding_dimensions = "[0, 1, 2]"

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %padded, %pad, %copy = transform.structured.pad %op pad_to_multiple_of {pad_to_multiple_of} {{\n"
            f"      padding_values = {padding_values},\n"
            f"      padding_dimensions = {padding_dimensions},\n"
            f'      copy_back_op = "linalg.copy"\n'
            f"    }} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %padded "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        return True

    # ------------------------------------------------------------------
    # RL parameter interface
    # ------------------------------------------------------------------

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.PAD_OPTIONS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        pad_multiple = cls.PAD_OPTIONS[raw_slots[0] % len(cls.PAD_OPTIONS)]
        return {"pad_multiple": pad_multiple}
