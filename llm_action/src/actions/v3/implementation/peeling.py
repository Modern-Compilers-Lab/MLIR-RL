from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Peeling(ActionBase):
    """
    Separate a loop into a main body with a trip count divisible by a given factor
    and a remainder loop handling leftover iterations.
    First tiles to create a loop, then peels that loop.
    Uses transform.structured.tile_using_for + transform.loop.peel.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "loop_index": {
                "description": "Index of the loop dimension to peel (0-based).",
                "type": "int",
                "default": 0,
            },
            "tile_size": {
                "description": "Tile size to create the loop to be peeled.",
                "type": "int",
                "default": None,
            },
            "peel_front": {
                "description": "If true, peel the first iteration; otherwise peel the last.",
                "type": "bool",
                "default": False,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        loop_index = params.get("loop_index", 0)
        if not isinstance(loop_index, int) or loop_index < 0:
            return False
        tile_size = params.get("tile_size")
        if tile_size is not None:
            if not isinstance(tile_size, int) or tile_size <= 0:
                return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        loop_index = params.get("loop_index", 0)
        tile_size = params.get("tile_size")
        peel_front = params.get("peel_front", False)

        if tile_size is None:
            tile_size = 16  # default tile size for peeling

        tile_sizes = [0] * (loop_index + 1)
        tile_sizes[loop_index] = tile_size

        peel_front_str = "true" if peel_front else "false"

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop0 = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)\n'
            f'    %main_loop, %remainder_loop = transform.loop.peel %loop0 {{peel_front = {peel_front_str}}} : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)\n'
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
