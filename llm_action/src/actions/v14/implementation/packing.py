from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Packing(ActionBase):
    """
    Pack the target linalg op operands into a blocked, contiguous layout using
    `transform.structured.pack`. The op's iteration space is blocked by
    `packed_sizes` so that inner-loop accesses become cache-friendly panels.

    Parameters:
      - packed_sizes: list[int], per-iteration-dim packing tile sizes.
        A value of 0 means the corresponding dim is not packed.
    """

    VOCAB = [0, 8, 16, 32, 64]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "description": (
                    "Per-iteration-dim packing tile sizes. 0 means 'do not "
                    "pack this dimension'."
                ),
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        sizes = params.get("packed_sizes")
        if not isinstance(sizes, (list, tuple)) or len(sizes) == 0:
            return False
        if not all(isinstance(s, int) and s >= 0 for s in sizes):
            return False
        if all(s == 0 for s in sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        sizes = list(params["packed_sizes"])
        sizes_str = "[" + ", ".join(str(int(s)) for s in sizes) + "]"

        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %op packed_sizes = {sizes_str} : (!transform.any_op) -> !transform.op<"linalg.generic">
    %tag = transform.param.constant "operation_0" -> !transform.any_param
    transform.annotate %packed "tag" = %tag : !transform.op<"linalg.generic">, !transform.any_param
    %pack_ops = transform.structured.match ops{{["linalg.pack"]}} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.pack">
    transform.structured.lower_pack %pack_ops : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
    %unpack_ops = transform.structured.match ops{{["linalg.unpack"]}} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.unpack">
    transform.structured.lower_unpack %unpack_ops : (!transform.op<"linalg.unpack">) -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">)
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
        return "linalg.pack" in after or "linalg.generic" in after

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
        return {"packed_sizes": sizes}
