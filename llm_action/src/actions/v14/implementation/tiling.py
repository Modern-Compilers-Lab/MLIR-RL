import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Tiling(ActionBase):
    """
    Partition the iteration space of the target linalg op loop band into smaller
    blocks using `transform.structured.tile_using_for`.

    Parameters:
      - tile_sizes: list[int], per-loop tile sizes. A value of 0 means that
        the corresponding loop is NOT tiled.
    """

    VOCAB = [0, 16, 32, 64, 128]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": (
                    "Per-loop tile sizes aligned with the loop band of the target "
                    "linalg op. 0 means 'do not tile this loop'."
                ),
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes")
        if not isinstance(tile_sizes, (list, tuple)):
            return False
        if len(tile_sizes) == 0:
            return False
        if not all(isinstance(t, int) and t >= 0 for t in tile_sizes):
            return False
        # Reject no-op: all zeros.
        if all(t == 0 for t in tile_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = list(params["tile_sizes"])
        n_loops = sum(1 for t in tile_sizes if t != 0)
        if n_loops == 0:
            return code
        result_types = ", ".join(["!transform.any_op"] * n_loops)
        sizes_str = "[" + ", ".join(str(int(t)) for t in tile_sizes) + "]"

        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {sizes_str} : (!transform.any_op) -> (!transform.any_op, {result_types})
    %tag = transform.param.constant "operation_0" -> !transform.any_param
    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param
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
        # Expect scf.for loops to be introduced.
        if "scf.for" not in after:
            return False
        return 'tag = "operation_0"' in after

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
