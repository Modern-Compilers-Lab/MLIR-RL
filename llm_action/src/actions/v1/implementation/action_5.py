from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class UnrollingAction(ActionBase):
    """
    Unrolling Action: Tiles the tagged operation with the given tile sizes,
    then unrolls the generated loops by the specified unroll factor.

    This effectively performs tiling followed by loop unrolling on the
    innermost generated loops, exposing instruction-level parallelism.

    Parameters:
        tile_sizes (list[int]): Tile sizes for each loop dimension. 0 means do not tile.
        unroll_factor (int): Number of loop body copies per iteration.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for each loop dimension before unrolling. 0 means do not tile.",
                "type": "list[int]",
                "values": None,
            },
            "unroll_factor": {
                "description": "Number of loop body copies per iteration.",
                "type": "int",
                "values": [2, 4, 8],
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False

        tile_sizes = params.get("tile_sizes", None)
        if tile_sizes is None or not isinstance(tile_sizes, list):
            return False
        if len(tile_sizes) == 0:
            return False
        if not all(isinstance(s, int) and s >= 0 for s in tile_sizes):
            return False
        if all(s == 0 for s in tile_sizes):
            return False

        unroll_factor = params.get("unroll_factor", None)
        if unroll_factor is None or not isinstance(unroll_factor, int):
            return False
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

        n_loops = sum(1 for s in tile_sizes if s != 0)
        loop_handles = ", ".join(["!transform.any_op"] * n_loops)

        # Name the loop results individually so we can unroll the innermost
        loop_names = [f"%loop{i}" for i in range(n_loops)]
        loop_names_str = ", ".join(loop_names)

        # Unroll the innermost loop (last generated loop)
        innermost_loop = loop_names[-1]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, {loop_names_str} = transform.structured.tile_using_for %op'
            f' tile_sizes {str(tile_sizes)} : (!transform.any_op) -> (!transform.any_op, {loop_handles})\n'
            f'    transform.loop.unroll {innermost_loop} {{factor = {unroll_factor}}}'
            f' : !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
            return result
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        if len(after.strip()) == 0:
            return False
        return True
