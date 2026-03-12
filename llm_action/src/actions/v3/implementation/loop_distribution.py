from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopDistribution(ActionBase):
    """
    Split a loop body containing multiple independent statements into separate loops,
    each executing one statement over the full iteration range.
    Uses transform.structured.tile_using_for to separate dimensions,
    effectively distributing the computation across separate loop nests.
    This action tiles with size 1 on the specified dimension, which separates
    the iteration space to enable independent processing.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "distribution_dimensions": {
                "description": "List of dimension indices to distribute (tile with size 1).",
                "type": "list[int]",
                "default": None,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        dist_dims = params.get("distribution_dimensions")
        if not dist_dims or not isinstance(dist_dims, list):
            return False
        if not all(isinstance(d, int) and d >= 0 for d in dist_dims):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        dist_dims = params["distribution_dimensions"]

        # Build tile_sizes: 1 at distribution dimensions, 0 elsewhere
        max_dim = max(dist_dims) + 1
        tile_sizes = [0] * max_dim
        for d in dist_dims:
            tile_sizes[d] = 1

        n_loops = len(dist_dims)
        loop_handles = ", ".join(["!transform.any_op"] * n_loops)
        loop_names = ", ".join([f"%loop{i}" for i in range(n_loops)])

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, {loop_names} = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_handles})\n'
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
