from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Packing(ActionBase):
    """
    Reorganize tensor data layout by introducing inner block dimensions that
    match tiled traversal order, converting strided accesses into contiguous
    accesses via linalg.pack/unpack.

    Structure-preserving (Category A): output is still a linalg.generic op
    (with pack/unpack wrappers).

    unique_execution = False because packing can be applied at different
    granularities (different packed_sizes).
    """

    unique_execution: bool = False  # can re-pack with different block sizes

    VOCAB = [0, 4, 8, 16, 32]  # per-dimension packed size; 0 = don't pack

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "description": "Inner block dimensions for packing along each loop dimension. "
                               "0 means do not pack that dimension.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        packed_sizes = params.get("packed_sizes", [])
        if not packed_sizes or not isinstance(packed_sizes, list):
            return False
        if all(s == 0 for s in packed_sizes):
            return False
        if any(not isinstance(s, int) or s < 0 for s in packed_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        packed_sizes = params["packed_sizes"]
        sizes_str = str(packed_sizes)

        # Pack the operation, then lower pack/unpack ops to standard tensor ops
        # so that the default bufferization pipeline can handle the result.
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %packed = transform.structured.pack %op packed_sizes = {sizes_str} : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %packed "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    %pack_ops = transform.structured.match ops{{["linalg.pack"]}} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.pack">\n'
            f'    %pad, %expand, %transpose = transform.structured.lower_pack %pack_ops : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)\n'
            f'    %unpack_ops = transform.structured.match ops{{["linalg.unpack"]}} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.unpack">\n'
            f'    %empty, %t2, %collapse, %extract = transform.structured.lower_unpack %unpack_ops : (!transform.op<"linalg.unpack">) -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">)\n'
            f'    %func = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %func : (!transform.any_op) -> !transform.any_op\n'
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

    @classmethod
    def params_size(cls) -> int:
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"packed_sizes": sizes}
