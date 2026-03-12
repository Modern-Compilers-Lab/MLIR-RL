from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Padding(ActionBase):
    """
    Extend tensor dimensions to ensure alignment with tile sizes, vector widths,
    or other hardware-required boundaries. Uses transform.structured.pad.
    Padding is typically applied after tiling to ensure tile dimensions are
    multiples of vector widths.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "padding_values": {
                "description": "List of padding values as float strings, one per operand of the linalg op (inputs + outputs).",
                "type": "list[float]",
                "default": None,
            },
            "padding_dimensions": {
                "description": "List of dimension indices to pad.",
                "type": "list[int]",
                "default": None,
            },
            "pad_to_multiple_of": {
                "description": "List of multiples to pad each dimension to. Must match length of padding_dimensions.",
                "type": "list[int]",
                "default": None,
            },
            "copy_back_op": {
                "description": "Strategy for copying back: 'bufferization.materialize_in_destination', 'linalg.copy', or 'none'.",
                "type": "str",
                "default": "bufferization.materialize_in_destination",
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        padding_dimensions = params.get("padding_dimensions")
        if not padding_dimensions or not isinstance(padding_dimensions, list):
            return False
        if not all(isinstance(d, int) and d >= 0 for d in padding_dimensions):
            return False
        padding_values = params.get("padding_values")
        if not padding_values or not isinstance(padding_values, list):
            return False
        copy_back_op = params.get("copy_back_op", "bufferization.materialize_in_destination")
        if copy_back_op not in ("bufferization.materialize_in_destination", "linalg.copy", "none"):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        padding_values = params["padding_values"]
        padding_dimensions = params["padding_dimensions"]
        pad_to_multiple_of = params.get("pad_to_multiple_of")
        copy_back_op = params.get("copy_back_op", "bufferization.materialize_in_destination")

        # Build padding_values attr: [0.0 : f64, 0.0 : f64, ...]
        pv_strs = []
        for v in padding_values:
            pv_strs.append(f"{float(v)} : f64")
        pv_attr = "[" + ", ".join(pv_strs) + "]"

        # Build padding_dimensions attr
        pd_attr = "[" + ", ".join(str(d) for d in padding_dimensions) + "]"

        pad_attrs = f'padding_values = {pv_attr}, padding_dimensions = {pd_attr}, copy_back_op = "{copy_back_op}"'

        if pad_to_multiple_of:
            ptm_list = "[" + ", ".join(str(m) for m in pad_to_multiple_of) + "]"
            pad_line = (
                f'    %padded, %pad, %copy = transform.structured.pad %op '
                f'{{{pad_attrs}}} '
                f'pad_to_multiple_of {ptm_list} '
                f': (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)'
            )
        else:
            pad_line = (
                f'    %padded, %pad, %copy = transform.structured.pad %op '
                f'{{{pad_attrs}}} '
                f': (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)'
            )

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'{pad_line}\n'
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
