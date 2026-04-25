import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Padding(ActionBase):
    """
    Pad the operands of the target linalg operation to ensure clean
    dimension sizes for downstream vectorization. Applied with
    padding_values and padding_dimensions.
    """

    VOCAB = [1, 4, 8, 16, 32]

    @classmethod
    def _detect_element_type(cls, code: str) -> str:
        m = re.search(r'tensor<[^>]*x(f16|f32|f64|bf16)>', code)
        return m.group(1) if m else "f64"

    @classmethod
    def parameters(cls) -> dict:
        return {
            "pad_to_multiple_of": {
                "description": "Padding multiple per dimension. 1 means no padding.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        pad_to = params.get("pad_to_multiple_of", [])
        if not pad_to or all(v == 1 for v in pad_to):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        pad_to = params["pad_to_multiple_of"]
        etype = cls._detect_element_type(code)
        n_operands = 3
        padding_values_str = ", ".join([f"0.0 : {etype}"] * n_operands)

        dims_to_pad = [i for i, v in enumerate(pad_to) if v > 1]
        if not dims_to_pad:
            return code
        padding_dims_str = ", ".join(str(d) for d in dims_to_pad)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %padded, %pad, %copy = transform.structured.pad %op'
            f' {{padding_values = [{padding_values_str}],'
            f' padding_dimensions = [{padding_dims_str}],'
            f' copy_back_op = "none"}}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %padded "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        values = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"pad_to_multiple_of": values}
