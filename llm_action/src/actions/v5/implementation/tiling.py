from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Tiling(ActionBase):
    """
    Partition the iteration space of a tagged linalg operation into
    multi-dimensional tiles using scf.for loops, so that each tile's
    data footprint fits within a target cache level.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "List of tile sizes, one per iteration dimension. 0 means do not tile that dimension.",
                "type": "list[int]",
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not isinstance(tile_sizes, list) or len(tile_sizes) == 0:
            return False
        if not all(isinstance(s, int) and s >= 0 for s in tile_sizes):
            return False
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

        loop_results = ", ".join(["!transform.any_op"] * n_loops)
        results_type = f"(!transform.any_op, {loop_results})"

        if n_loops > 1:
            binding = f"%tiled_op, %loops:{n_loops}"
        else:
            binding = "%tiled_op, %loop"

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg0 : (!transform.any_op) -> !transform.any_op\n'
            f'    {binding} = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> {results_type}\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
