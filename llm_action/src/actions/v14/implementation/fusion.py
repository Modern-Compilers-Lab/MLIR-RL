from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Fusion(ActionBase):
    """
    Tile the target linalg op and greedily fuse any producer ops into the
    resulting tile loops using `transform.structured.fuse`. For a single
    payload op with no producers, this degrades to plain tiling but still
    produces a valid fused loop nest that later producers can be fused into.

    Parameters:
      - tile_sizes: list[int], per-loop tile sizes used as fusion anchors.
        A value of 0 means 'do not tile/fuse this loop'.
    """

    VOCAB = [0, 8, 16, 32, 64]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": (
                    "Per-loop tile sizes used by structured.fuse to create "
                    "fusion anchors. A value of 0 means 'do not tile/fuse "
                    "this loop'."
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
        n_loops = sum(1 for s in sizes if s > 0)
        if n_loops == 0:
            return code
        sizes_str = "[" + ", ".join(str(int(s)) for s in sizes) + "]"
        loop_results = ", ".join([f"%loop{i}" for i in range(n_loops)])
        loop_types = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused, {loop_results} = transform.structured.fuse %op {sizes_str} : (!transform.any_op) -> (!transform.any_op, {loop_types})
    %tag = transform.param.constant "operation_0" -> !transform.any_param
    transform.annotate %fused "tag" = %tag : !transform.any_op, !transform.any_param
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
        # Fusion emits scf.for loops around the fused payload.
        return after.count("scf.for") > before.count("scf.for")

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
