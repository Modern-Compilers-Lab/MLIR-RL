from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class TilingAction(ActionBase):
    """
    Tiling Action: Partitions the iteration space of a tagged linalg operation
    into smaller blocks (tiles) using scf.for loops, so that the data footprint
    of each tile fits within a target cache level.

    Parameters:
        tile_sizes (list[int]): Tile sizes for each loop dimension. A value of 0
            means "do not tile that dimension". Length must match the number of
            loops in the target operation.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for each loop dimension. 0 means do not tile.",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        # Check tag exists
        if 'tag = "operation_0"' not in code:
            return False

        tile_sizes = params.get("tile_sizes", None)
        if tile_sizes is None or not isinstance(tile_sizes, list):
            return False

        if len(tile_sizes) == 0:
            return False

        # All elements must be non-negative integers
        if not all(isinstance(s, int) and s >= 0 for s in tile_sizes):
            return False

        # At least one non-zero tile size (otherwise it's a no-op)
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
        loop_handles = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op'
            f' tile_sizes {str(tile_sizes)} : (!transform.any_op) -> (!transform.any_op, {loop_handles})\n'
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
