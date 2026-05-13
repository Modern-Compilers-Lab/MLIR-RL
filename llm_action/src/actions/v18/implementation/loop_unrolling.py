from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopUnrolling(ActionBase):
    """
    Replicate the loop body multiple times within a single iteration by first
    tiling the operation (to produce scf.for loops) and then unrolling the
    innermost loop by the given factor.
    """

    TILE_VOCAB = [4, 8, 16, 32, 64]
    UNROLL_VOCAB = [2, 4, 8]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_size": {
                "description": "Tile size for the innermost loop dimension (used to create the loop to unroll).",
                "type": "int",
                "values": cls.TILE_VOCAB,
            },
            "unroll_factor": {
                "description": "The number of loop body copies per iteration.",
                "type": "int",
                "values": cls.UNROLL_VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_size = params.get("tile_size")
        unroll_factor = params.get("unroll_factor")
        if tile_size is None or unroll_factor is None:
            return False
        if not isinstance(tile_size, int) or tile_size <= 0:
            return False
        if not isinstance(unroll_factor, int) or unroll_factor <= 1:
            return False
        if tile_size < unroll_factor:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_size = params["tile_size"]
        unroll_factor = params["unroll_factor"]

        # Tile only the innermost (reduction/K) dimension, then unroll its loop
        tile_sizes = [0, 0, tile_size]

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f"    %op = transform.structured.match attributes{{tag = \"operation_0\"}} in %arg1"
            f" : (!transform.any_op) -> !transform.any_op\n"
            f"    %tiled_op, %loop = transform.structured.tile_using_for %op"
            f" tile_sizes {str(tile_sizes)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
            f"    %tag = transform.param.constant \"operation_0\" -> !transform.any_param\n"
            f"    transform.annotate %tiled_op \"tag\" = %tag : !transform.any_op, !transform.any_param\n"
            f"    transform.loop.unroll %loop {{factor = {unroll_factor}}} : !transform.any_op\n"
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
        return 2

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.TILE_VOCAB), len(cls.UNROLL_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        tile_size = cls.TILE_VOCAB[raw_slots[0] % len(cls.TILE_VOCAB)]
        unroll_factor = cls.UNROLL_VOCAB[raw_slots[1] % len(cls.UNROLL_VOCAB)]
        # Ensure tile_size >= unroll_factor
        if tile_size < unroll_factor:
            unroll_factor = tile_size
        return {"tile_size": tile_size, "unroll_factor": unroll_factor}
