from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """
    Promote operand sub-tensors accessed within a tile into contiguous
    temporary buffers (alloca) before computation, ensuring conflict-free
    cache-line access. This action tiles the operation first, then promotes
    specified operands of the tiled inner operation.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes to create the tile scope. 0 means do not tile that dimension.",
                "type": "list[int]",
                "default": None,
            },
            "operands_to_promote": {
                "description": "List of operand indices to promote (0-based). E.g. [0, 1] promotes both inputs.",
                "type": "list[int]",
                "default": None,
            },
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
        operands = params.get("operands_to_promote")
        if not operands or not isinstance(operands, list):
            return False
        if not all(isinstance(o, int) and o >= 0 for o in operands):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        operands = params["operands_to_promote"]

        n_loops = sum(1 for s in tile_sizes if s != 0)
        loop_types = ", ".join(["!transform.any_op"] * n_loops)
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op'
            f' tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_types})\n'
            f'    %promoted_op = transform.structured.promote %tiled_op'
            f' : (!transform.any_op) -> !transform.any_op\n'
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
