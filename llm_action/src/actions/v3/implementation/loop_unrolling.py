from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopUnrolling(ActionBase):
    """
    Replicate the loop body multiple times, reducing loop overhead and exposing
    independent instructions for pipelining. First tiles the target operation to
    produce a loop, then unrolls that loop by the given factor.
    Uses transform.structured.tile_using_for + transform.loop.unroll.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "unroll_factor": {
                "description": "The number of loop body copies per iteration.",
                "type": "int",
                "default": None,
            },
            "loop_index": {
                "description": "Index of the loop dimension to unroll (0-based). The dimension must be tileable.",
                "type": "int",
                "default": 0,
            },
            "tile_size": {
                "description": "Tile size to create the loop to be unrolled. Must be divisible by unroll_factor for clean unrolling.",
                "type": "int",
                "default": None,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        unroll_factor = params.get("unroll_factor")
        if not unroll_factor or not isinstance(unroll_factor, int) or unroll_factor < 2:
            return False
        loop_index = params.get("loop_index", 0)
        if not isinstance(loop_index, int) or loop_index < 0:
            return False
        tile_size = params.get("tile_size")
        if tile_size is not None:
            if not isinstance(tile_size, int) or tile_size < unroll_factor:
                return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        unroll_factor = params["unroll_factor"]
        loop_index = params.get("loop_index", 0)
        tile_size = params.get("tile_size", unroll_factor)
        if tile_size is None:
            tile_size = unroll_factor

        # Build tile_sizes: tile_size at loop_index, 0 elsewhere
        # We don't know n_dims; we generate tile_sizes dynamically
        # Use a large enough list with zeros padded
        # Actually, we need exactly the right number of dims. We'll use
        # a variable number approach: tile only the target dimension.
        # We use the fact that tile_sizes with fewer entries than dims
        # are zero-padded by MLIR.
        tile_sizes = [0] * (loop_index + 1)
        tile_sizes[loop_index] = tile_size

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop0 = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    transform.loop.unroll %loop0 {{factor = {unroll_factor}}} : !transform.any_op\n'
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
