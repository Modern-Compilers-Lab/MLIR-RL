from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Fusion(ActionBase):
    """Tiles the target operation and greedily fuses producer operations into
    the tiled loop using transform.structured.fuse.

    The fuse transform returns 1 + N handles where N is the number of non-zero
    tile sizes: the first handle is the fused operation, and the remaining
    handles correspond to the generated loops.
    """

    VOCAB = [0, 4, 8, 16, 32]

    # ------------------------------------------------------------------
    # ActionBase interface
    # ------------------------------------------------------------------

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": (
                    "List of tile sizes for fusion tiling per loop dimension. "
                    "0 means do not tile that dimension."
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
        if not tile_sizes or not isinstance(tile_sizes, list):
            return False
        if not all(isinstance(s, int) and s >= 0 for s in tile_sizes):
            return False
        if all(s == 0 for s in tile_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_loops = sum(1 for s in tile_sizes if s != 0)
        n_results = 1 + n_loops
        result_types = ", ".join(["!transform.any_op"] * n_results)

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %fused:{n_results} = transform.structured.fuse %op {{tile_sizes = {tile_sizes}}} : (!transform.any_op) -> ({result_types})\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %fused#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"tile_sizes": sizes}
