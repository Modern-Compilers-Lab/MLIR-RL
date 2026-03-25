from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Tiling(ActionBase):
    """
    Partitions the iteration space of a tagged linalg operation into smaller
    tiles using scf.for loops, improving data locality for cache-friendly access.

    Uses transform.structured.tile_using_for on the operation matched by
    tag = "operation_0". Re-annotates the tiled inner operation with the tag.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "List of tile sizes, one per loop dimension. 0 means do not tile that dimension.",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or all(s == 0 for s in tile_sizes):
            return False
        if any(s < 0 for s in tile_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_loops = sum(1 for s in tile_sizes if s != 0)
        loop_results = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main("
            f"%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} '
            f"in %arg1 : (!transform.any_op) -> !transform.any_op\n"
            f"    %tiled_op, %loops:{n_loops} = "
            f"transform.structured.tile_using_for %op tile_sizes {tile_sizes} "
            f": (!transform.any_op) -> (!transform.any_op, {loop_results})\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f"    transform.annotate %tiled_op "
            f'"tag" = %tag : !transform.any_op, !transform.any_param\n'
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
