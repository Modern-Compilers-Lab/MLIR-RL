from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Peeling(ActionBase):
    """
    Two-level tiling: first tile the operation with the given outer tile sizes,
    then tile the inner operation again with a fixed micro-tile factor.
    This creates a multi-level loop nest suitable for exploiting multiple
    cache hierarchy levels (e.g. L2 blocking + L1/register blocking).
    """

    VOCAB = [0, 4, 8, 16, 32]
    INNER_FACTOR = 4  # Fixed micro-tile factor for the second level

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Outer tile sizes per iterator dimension. 0 means no tiling.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or all(s == 0 for s in tile_sizes):
            return False
        # Need at least one tile size > INNER_FACTOR for two-level tiling to be meaningful
        if not any(s > cls.INNER_FACTOR for s in tile_sizes if s != 0):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_loops = sum(1 for s in tile_sizes if s != 0)
        if n_loops == 0:
            return code

        loop_types = ", ".join(["!transform.any_op"] * n_loops)
        tile_sizes_str = str(tile_sizes).replace("'", "")

        # Compute inner tile sizes: for each non-zero outer size, use INNER_FACTOR
        # if the outer size is large enough, otherwise 0 (skip)
        inner_sizes = [
            cls.INNER_FACTOR if s >= cls.INNER_FACTOR * 2 else 0
            for s in tile_sizes
        ]
        n_inner_loops = sum(1 for s in inner_sizes if s != 0)

        if n_inner_loops == 0:
            # Fall back to single-level tiling if inner sizes are all zero
            inner_sizes = [cls.INNER_FACTOR if s != 0 else 0 for s in tile_sizes]
            n_inner_loops = sum(1 for s in inner_sizes if s != 0)

        inner_loop_types = ", ".join(["!transform.any_op"] * n_inner_loops)
        inner_sizes_str = str(inner_sizes).replace("'", "")

        # First level: outer tiling
        if n_loops > 1:
            outer_tile = (
                f'    %tiled_op1, %loops1:{n_loops} = transform.structured.tile_using_for %op'
                f' tile_sizes {tile_sizes_str}'
                f' : (!transform.any_op) -> (!transform.any_op, {loop_types})\n'
            )
        else:
            outer_tile = (
                f'    %tiled_op1, %loops1 = transform.structured.tile_using_for %op'
                f' tile_sizes {tile_sizes_str}'
                f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            )

        # Second level: inner tiling (micro-tiling)
        if n_inner_loops > 1:
            inner_tile = (
                f'    %tiled_op2, %loops2:{n_inner_loops} = transform.structured.tile_using_for %inner'
                f' tile_sizes {inner_sizes_str}'
                f' : (!transform.any_op) -> (!transform.any_op, {inner_loop_types})\n'
            )
        else:
            inner_tile = (
                f'    %tiled_op2, %loops2 = transform.structured.tile_using_for %inner'
                f' tile_sizes {inner_sizes_str}'
                f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            )

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            + outer_tile +
            f'    %tag1 = transform.param.constant "inner_tile" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op1 "tag" = %tag1 : !transform.any_op, !transform.any_param\n'
            f'    %inner = transform.structured.match attributes{{tag = "inner_tile"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            + inner_tile +
            f'    %tag2 = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op2 "tag" = %tag2 : !transform.any_op, !transform.any_param\n'
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
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"tile_sizes": sizes}
