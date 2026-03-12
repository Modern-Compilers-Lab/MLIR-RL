from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """
    Copy a sub-tensor accessed within a tile into a contiguous temporary buffer
    (scratchpad) before computation. In the tensor world, this is achieved by
    tiling the operation and then padding the tiled operands, which creates
    contiguous temporary tensors that serve as promoted buffers after bufferization.
    Uses transform.structured.tile_using_for + transform.structured.pad.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes to apply before promotion. Promotion requires tiled operands.",
                "type": "list[int]",
                "default": None,
            },
            "padding_values": {
                "description": "Padding values as floats, one per operand (inputs + outputs).",
                "type": "list[float]",
                "default": None,
            },
            "padding_dimensions": {
                "description": "List of dimension indices to pad after tiling.",
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
        padding_values = params.get("padding_values")
        if not padding_values or not isinstance(padding_values, list):
            return False
        padding_dimensions = params.get("padding_dimensions")
        if not padding_dimensions or not isinstance(padding_dimensions, list):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        padding_values = params["padding_values"]
        padding_dimensions = params["padding_dimensions"]

        n_loops = sum(1 for s in tile_sizes if s != 0)
        loop_handles = ", ".join(["!transform.any_op"] * n_loops)
        loop_names = ", ".join([f"%loop{i}" for i in range(n_loops)])

        pv_strs = [f"{float(v)} : f64" for v in padding_values]
        pv_attr = "[" + ", ".join(pv_strs) + "]"
        pd_attr = "[" + ", ".join(str(d) for d in padding_dimensions) + "]"

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, {loop_names} = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_handles})\n'
            f'    %padded, %pad, %copy = transform.structured.pad %tiled_op {{padding_values = {pv_attr}, padding_dimensions = {pd_attr}, copy_back_op = "none"}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n'
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
