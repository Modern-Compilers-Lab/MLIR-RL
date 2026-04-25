from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Padding(ActionBase):
    """
    Pad the iteration space of the target linalg op so that each padded
    dimension becomes a multiple of a hardware-friendly constant. We use
    `transform.structured.pad` which inserts `tensor.pad` ops on the operands
    and copies the relevant slice of the result back to the original
    destination tensor.

    Parameters:
      - pad_to_multiple_of: list[int], per-loop multiples. A value of 1 means
        "do not pad this dimension".
    """

    VOCAB = [1, 16, 32, 48, 64]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "pad_to_multiple_of": {
                "description": (
                    "Per-loop multiples to pad the iteration space to. A "
                    "value of 1 means 'no padding on this dimension'. "
                    "Static dimensions already divisible by the value result "
                    "in a no-op pad."
                ),
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        sizes = params.get("pad_to_multiple_of")
        if not isinstance(sizes, (list, tuple)) or len(sizes) == 0:
            return False
        if not all(isinstance(s, int) and s >= 1 for s in sizes):
            return False
        # At least one dimension must have a non-trivial pad target.
        if all(s <= 1 for s in sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        sizes = list(params["pad_to_multiple_of"])
        n = len(sizes)
        sizes_str = "[" + ", ".join(str(int(s)) for s in sizes) + "]"
        dims_str = "[" + ", ".join(str(i) for i in range(n)) + "]"
        padding_values = "[" + ", ".join(["0.0 : f64"] * 3) + "]"

        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %copy_back = transform.structured.pad %op pad_to_multiple_of {sizes_str} {{
      padding_values = {padding_values},
      padding_dimensions = {dims_str},
      copy_back_op = "none"
    }} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tag = transform.param.constant "operation_0" -> !transform.any_param
    transform.annotate %padded "tag" = %tag : !transform.any_op, !transform.any_param
    transform.yield
  }}
}}
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
        # Padding should introduce a tensor.pad op on at least one operand.
        return "tensor.pad" in after

    @classmethod
    def params_size(cls) -> int:
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(max(n_loops, 1), MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(max(n_loops, 1), MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"pad_to_multiple_of": sizes}
