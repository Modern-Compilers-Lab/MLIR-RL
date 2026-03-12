from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class MultiLevelTiling(ActionBase):
    """
    Apply hierarchical tiling with multiple tile size levels, producing nested tile
    loops corresponding to different cache levels (e.g., L2 tiles containing L1 tiles).
    Applies two successive rounds of transform.structured.tile_using_for.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes_l2": {
                "description": "List of tile sizes for the outer (L2) tiling level. 0 means do not tile that dimension.",
                "type": "list[int]",
                "default": None,
            },
            "tile_sizes_l1": {
                "description": "List of tile sizes for the inner (L1) tiling level. 0 means do not tile that dimension.",
                "type": "list[int]",
                "default": None,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        for key in ("tile_sizes_l2", "tile_sizes_l1"):
            ts = params.get(key)
            if not ts or not isinstance(ts, list):
                return False
            if not all(isinstance(s, int) and s >= 0 for s in ts):
                return False
            if all(s == 0 for s in ts):
                return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        ts_l2 = params["tile_sizes_l2"]
        ts_l1 = params["tile_sizes_l1"]

        n_loops_l2 = sum(1 for s in ts_l2 if s != 0)
        loop_handles_l2 = ", ".join(["!transform.any_op"] * n_loops_l2)
        loop_names_l2 = ", ".join([f"%l2_loop{i}" for i in range(n_loops_l2)])

        n_loops_l1 = sum(1 for s in ts_l1 if s != 0)
        loop_handles_l1 = ", ".join(["!transform.any_op"] * n_loops_l1)
        loop_names_l1 = ", ".join([f"%l1_loop{i}" for i in range(n_loops_l1)])

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %l2_tiled, {loop_names_l2} = transform.structured.tile_using_for %op tile_sizes {ts_l2} : (!transform.any_op) -> (!transform.any_op, {loop_handles_l2})\n'
            f'    %l1_tiled, {loop_names_l1} = transform.structured.tile_using_for %l2_tiled tile_sizes {ts_l1} : (!transform.any_op) -> (!transform.any_op, {loop_handles_l1})\n'
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
        # Should have nested scf.for loops
        if "scf.for" not in after:
            return False
        return True
