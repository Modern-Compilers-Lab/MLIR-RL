from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """
    Materialize a sub-tile of the target linalg op into a freshly allocated
    local buffer. We first tile the op so the inner op operates on a sub-slice,
    then use `transform.structured.bufferize_to_allocation` to promote the
    destination operand to a local allocation.

    Parameters:
      - tile_sizes: list[int], per-loop tile sizes used to create the inner
        sub-tile that is promoted. A value of 0 means no tiling for that loop.
    """

    VOCAB = [0, 8, 16, 32, 64]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": (
                    "Per-loop tile sizes used to create the inner sub-tile "
                    "that will be promoted. 0 means 'do not tile this loop'."
                ),
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        sizes = params.get("tile_sizes")
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
        sizes = list(params["tile_sizes"])
        n_loops = sum(1 for t in sizes if t != 0)
        if n_loops == 0:
            return code
        result_types = ", ".join(["!transform.any_op"] * n_loops)
        sizes_str = "[" + ", ".join(str(int(s)) for s in sizes) + "]"

        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {sizes_str} : (!transform.any_op) -> (!transform.any_op, {result_types})
    %buf, %new_ops = transform.structured.bufferize_to_allocation %tiled_op {{bufferize_destination_only}} : !transform.any_op
    %matched = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tag = transform.param.constant "operation_0" -> !transform.any_param
    transform.annotate %matched "tag" = %tag : !transform.any_op, !transform.any_param
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
        # Expect an allocation or to_tensor materialization.
        return "memref.alloc" in after or "bufferization.to_tensor" in after

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
        return {"tile_sizes": sizes}
