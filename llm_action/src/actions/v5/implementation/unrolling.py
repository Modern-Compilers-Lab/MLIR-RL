from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """
    Unroll a loop in the IR by a given factor. First tiles the tagged
    operation to create a loop, then unrolls that loop.
    The target loop is selected by tiling dimension index.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_size": {
                "description": "Tile size for the target dimension to create the loop to unroll.",
                "type": "int",
            },
            "dimension": {
                "description": "Which iteration dimension to tile (0-indexed).",
                "type": "int",
            },
            "unroll_factor": {
                "description": "Number of iterations to unroll.",
                "type": "int",
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_size = params.get("tile_size", 0)
        dimension = params.get("dimension", -1)
        unroll_factor = params.get("unroll_factor", 0)
        if not isinstance(tile_size, int) or tile_size <= 0:
            return False
        if not isinstance(dimension, int) or dimension < 0:
            return False
        if not isinstance(unroll_factor, int) or unroll_factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_size = params["tile_size"]
        dimension = params["dimension"]
        unroll_factor = params["unroll_factor"]

        # Build tile_sizes list with only the target dimension non-zero
        # We need enough entries; use dimension+1 as minimum
        tile_sizes = [0] * (dimension + 1)
        tile_sizes[dimension] = tile_size

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg0 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
