from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class PackingAction(ActionBase):
    """
    Packing Action: Applies data tiling (packing) to a tagged linalg operation
    using transform.structured.pack, reorganizing operand data into contiguous
    tiles with computation-friendly layouts.

    Parameters:
        packed_sizes (list[int]): Pack sizes for each iterator dimension.
            0 means "do not pack that dimension".
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "description": "Pack sizes for each iterator dimension. 0 means do not pack.",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False

        packed_sizes = params.get("packed_sizes", None)
        if packed_sizes is None or not isinstance(packed_sizes, list):
            return False

        if len(packed_sizes) == 0:
            return False

        if not all(isinstance(s, int) and s >= 0 for s in packed_sizes):
            return False

        # At least one non-zero size
        if all(s == 0 for s in packed_sizes):
            return False

        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        packed_sizes = params["packed_sizes"]
        sizes_str = "[" + ", ".join(str(s) for s in packed_sizes) + "]"

        # Pack the target op, then lower pack/unpack ops so the result
        # can be bufferized and executed by the standard pipeline.
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %packed_op = transform.structured.pack %op'
            f' packed_sizes = {sizes_str}'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %pack_ops = transform.structured.match ops{{["linalg.pack"]}} in %arg1'
            f' : (!transform.any_op) -> !transform.op<"linalg.pack">\n'
            f'    %pad, %expand, %transpose = transform.structured.lower_pack %pack_ops'
            f' : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)\n'
            f'    %unpack_ops = transform.structured.match ops{{["linalg.unpack"]}} in %arg1'
            f' : (!transform.any_op) -> !transform.op<"linalg.unpack">\n'
            f'    %empty, %t2, %collapse, %extract = transform.structured.lower_unpack %unpack_ops'
            f' : (!transform.op<"linalg.unpack">) -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">)\n'
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
