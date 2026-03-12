from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Peeling(ActionBase):
    """
    Split a loop into a main body with a trip count divisible by a given
    factor and a remainder loop handling leftover iterations. This is done
    by first tiling the operation to create loops, then peeling the
    innermost generated loop.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes to use before peeling. At least one must be non-zero "
                               "and should NOT evenly divide the corresponding dimension for peeling to have effect.",
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
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_loops = sum(1 for s in tile_sizes if s != 0)
        loop_types = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op'
            f' tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_types})\n'
            f'    %parent = transform.get_parent_op %tiled_op {{op_name = "scf.for"}}'
            f' : (!transform.any_op) -> !transform.op<"scf.for">\n'
            f'    %main, %remainder = transform.loop.peel %parent {{peel_front = false}}'
            f' : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)\n'
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
