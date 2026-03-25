from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Packing(ActionBase):
    """
    Pack operands of a tagged linalg operation into contiguous tiles
    with optimized layout, eliminating strided accesses.
    Uses transform.structured.pack with specified packed_sizes,
    then lowers pack/unpack ops for downstream bufferization compatibility.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "description": "List of packed tile sizes, one per iteration dimension. 0 means do not pack that dimension.",
                "type": "list[int]",
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        packed_sizes = params.get("packed_sizes", [])
        if not isinstance(packed_sizes, list) or len(packed_sizes) == 0:
            return False
        if not all(isinstance(s, int) and s >= 0 for s in packed_sizes):
            return False
        if all(s == 0 for s in packed_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        packed_sizes = params["packed_sizes"]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg0 : (!transform.any_op) -> !transform.any_op\n'
            f'    %packed_op = transform.structured.pack %op packed_sizes = {packed_sizes} : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %packed_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'\n'
            f'    %pack_ops = transform.structured.match ops{{["linalg.pack"]}} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.pack">\n'
            f'    transform.structured.lower_pack %pack_ops : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)\n'
            f'\n'
            f'    %unpack_ops = transform.structured.match ops{{["linalg.unpack"]}} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.unpack">\n'
            f'    transform.structured.lower_unpack %unpack_ops : (!transform.op<"linalg.unpack">) -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">)\n'
            f'\n'
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
