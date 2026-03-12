from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Fusion(ActionBase):
    """
    Fusion action: tiles the target operation and fuses its producers greedily
    into the generated loop nest. Uses transform.structured.fuse which combines
    tiling and producer fusion in a single step.

    This is particularly useful after transforms like packing that introduce
    producer ops (linalg.pack), enabling them to be fused into the compute loops.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "List of tile sizes for the fusion. One per loop dimension. 0 means do not tile that dimension. At least one dimension must be non-zero to create a loop nest for fusion.",
                "type": "list[int]",
                "values": None,
            }
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
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %fused:{n_loops + 1} = transform.structured.fuse %op {tile_sizes}'
            f' : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
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
