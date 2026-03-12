from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopFusionAction(ActionBase):
    """
    Loop Fusion Action: Tiles the target tagged operation and greedily fuses
    its producer operations into the generated loop nest using
    transform.structured.fuse.

    This is "tile-and-fuse": it tiles the consumer and pulls producers
    into the tiled loops, reducing intermediate memory traffic.

    Parameters:
        tile_sizes (list[int]): Tile sizes for the consumer operation.
            A value of 0 means "do not tile that dimension".
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for the consumer op during tile-and-fuse. 0 means do not tile.",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False

        tile_sizes = params.get("tile_sizes", None)
        if tile_sizes is None or not isinstance(tile_sizes, list):
            return False

        if len(tile_sizes) == 0:
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
        total_results = 1 + n_loops  # fused_op + loop handles
        result_types = ", ".join(["!transform.any_op"] * total_results)
        result_names = ", ".join([f"%r{i}" for i in range(total_results)])

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    {result_names} = transform.structured.fuse %op'
            f' {str(tile_sizes)}'
            f' : (!transform.any_op) -> ({result_types})\n'
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
