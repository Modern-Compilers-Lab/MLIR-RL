from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """
    Unrolling action: tiles the target operation to create a loop, then unrolls
    that loop by the specified factor. This reduces loop overhead and exposes
    instruction-level parallelism.

    The action first tiles the target op along a specified dimension to create
    an scf.for loop, then applies transform.loop.unroll on that loop.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_size": {
                "description": "Tile size for the dimension to unroll. Creates a loop with this trip count per tile.",
                "type": "int",
                "values": None,
            },
            "unroll_factor": {
                "description": "Number of times to replicate the loop body. Must be a positive integer.",
                "type": "int",
                "values": None,
            },
            "dimension": {
                "description": "Index of the dimension to tile and unroll (0-based).",
                "type": "int",
                "values": None,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_size = params.get("tile_size")
        unroll_factor = params.get("unroll_factor")
        dimension = params.get("dimension")
        if not isinstance(tile_size, int) or tile_size <= 0:
            return False
        if not isinstance(unroll_factor, int) or unroll_factor <= 1:
            return False
        if not isinstance(dimension, int) or dimension < 0:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_size = params["tile_size"]
        unroll_factor = params["unroll_factor"]
        dimension = params["dimension"]

        tile_sizes = [0] * dimension + [tile_size]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop = transform.structured.tile_using_for %op'
            f' tile_sizes {tile_sizes}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    transform.loop.unroll %loop {{factor = {unroll_factor}}} : !transform.any_op\n'
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
