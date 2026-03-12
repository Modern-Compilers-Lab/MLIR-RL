from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Padding(ActionBase):
    """
    Extend tensor dimensions to multiples of tile sizes, vector widths,
    or other alignment boundaries, ensuring regular tile shapes and clean
    vectorization. Uses transform.structured.pad on the tagged operation.
    Requires tensor semantics (not memref).
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "padding_values": {
                "description": "Padding values for each operand (as float strings). "
                               "E.g. ['0.0', '0.0', '0.0'] for 3 operands.",
                "type": "list[str]",
                "default": None,
            },
            "padding_dimensions": {
                "description": "Which dimensions to pad (list of dimension indices).",
                "type": "list[int]",
                "default": None,
            },
            "pack_paddings": {
                "description": "Which operands to pack (1 = pack, 0 = don't). "
                               "E.g. [1, 1, 1] packs all operands.",
                "type": "list[int]",
                "default": None,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        # Padding requires tensor semantics
        if "memref<" in code and "tensor<" not in code:
            return False
        pv = params.get("padding_values")
        if not pv or not isinstance(pv, list):
            return False
        pd = params.get("padding_dimensions")
        if not pd or not isinstance(pd, list):
            return False
        if not all(isinstance(d, int) and d >= 0 for d in pd):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        padding_values = params["padding_values"]
        padding_dimensions = params["padding_dimensions"]
        pack_paddings = params.get("pack_paddings")

        pv_str = ", ".join(f"{v} : f64" for v in padding_values)
        pd_str = ", ".join(str(d) for d in padding_dimensions)

        attrs = f"padding_values = [{pv_str}], padding_dimensions = [{pd_str}]"
        if pack_paddings:
            pp_str = ", ".join(str(p) for p in pack_paddings)
            attrs += f", pack_paddings = [{pp_str}]"

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %padded, %pad_op, %copy_back = transform.structured.pad %op'
            f' {{{attrs}}}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n'
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
