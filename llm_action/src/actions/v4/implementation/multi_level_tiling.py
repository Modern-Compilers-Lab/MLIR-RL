from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class MultiLevelTiling(ActionBase):
    """
    Apply hierarchical tiling at two levels, producing nested tile loops
    targeting different levels of the memory hierarchy (e.g., L2 outer tiles
    containing L1 inner tiles).
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "outer_tile_sizes": {
                "description": "Tile sizes for the outer (L2) tiling level. 0 means do not tile.",
                "type": "list[int]",
                "default": None,
            },
            "inner_tile_sizes": {
                "description": "Tile sizes for the inner (L1) tiling level. 0 means do not tile.",
                "type": "list[int]",
                "default": None,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        for key in ("outer_tile_sizes", "inner_tile_sizes"):
            sizes = params.get(key)
            if not sizes or not isinstance(sizes, list):
                return False
            if not all(isinstance(s, int) and s >= 0 for s in sizes):
                return False
            if all(s == 0 for s in sizes):
                return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        outer_sizes = params["outer_tile_sizes"]
        inner_sizes = params["inner_tile_sizes"]

        n_outer = sum(1 for s in outer_sizes if s != 0)
        n_inner = sum(1 for s in inner_sizes if s != 0)
        outer_loop_types = ", ".join(["!transform.any_op"] * n_outer)
        inner_loop_types = ", ".join(["!transform.any_op"] * n_inner)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %outer_op, %outer_loops:{n_outer} = transform.structured.tile_using_for %op'
            f' tile_sizes {outer_sizes} : (!transform.any_op) -> (!transform.any_op, {outer_loop_types})\n'
            f'    %inner_op, %inner_loops:{n_inner} = transform.structured.tile_using_for %outer_op'
            f' tile_sizes {inner_sizes} : (!transform.any_op) -> (!transform.any_op, {inner_loop_types})\n'
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
