from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """Tile the loop nest and fully unroll the innermost generated loop to expose ILP.

    Structure-preserving (Category A): the linalg op copies persist after unrolling.
    Tags the outermost tile loop as the computation entry point.
    """

    unique_execution: bool = True  # Can unroll different loops at different tile levels

    TILE_VOCAB = [4, 8, 16, 32]
    UNROLL_VOCAB = [2, 4, 8]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for the loop nest before unrolling. All must be > 0.",
                "type": "list[int]",
                "values": cls.TILE_VOCAB,
            },
            "unroll_factor": {
                "description": "Unroll factor for the innermost generated loop.",
                "type": "int",
                "values": cls.UNROLL_VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes:
            return False
        if any(s <= 0 for s in tile_sizes):
            return False
        unroll_factor = params.get("unroll_factor", 0)
        if unroll_factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        unroll_factor = params["unroll_factor"]
        n_loops = len(tile_sizes)
        r = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f" : (!transform.any_op) -> !transform.any_op\n"
            f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op"
            f" tile_sizes {str(tile_sizes)} : (!transform.any_op) -> (!transform.any_op, {r})\n"
            # Replace inherited tag on tiled_op to avoid duplicate "operation_0" after unrolling
            f'    %inner_tag = transform.param.constant "unrolled_inner" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %inner_tag : !transform.any_op, !transform.any_param\n'
            # Tag outermost loop as the computation entry point
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
            # Unroll the innermost loop
            f"    transform.loop.unroll %loops#{n_loops - 1} {{factor = {unroll_factor}}}"
            f" : !transform.any_op\n"
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

    @classmethod
    def params_size(cls) -> int:
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        n = min(n_loops, MAX_PARAM_SLOTS - 1)
        return [len(cls.TILE_VOCAB)] * n + [len(cls.UNROLL_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS - 1)
        tile_sizes = [cls.TILE_VOCAB[raw_slots[i] % len(cls.TILE_VOCAB)] for i in range(n)]
        unroll_factor = cls.UNROLL_VOCAB[raw_slots[n] % len(cls.UNROLL_VOCAB)]
        return {"tile_sizes": tile_sizes, "unroll_factor": unroll_factor}
