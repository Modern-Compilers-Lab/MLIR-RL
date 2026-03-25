from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Packing(ActionBase):
    """Packs a tagged linalg operation using transform.structured.pack.

    Copies sub-tiles of operand data into contiguous scratch buffers,
    converting strided accesses into sequential unit-stride reads.
    """

    VOCAB = [0, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "description": "List of packed sizes per iterator dimension. 0 means don't pack that dimension.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        packed_sizes = params.get("packed_sizes")
        if not packed_sizes or not isinstance(packed_sizes, list):
            return False
        if not all(isinstance(s, int) and s >= 0 for s in packed_sizes):
            return False
        if all(s == 0 for s in packed_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    LOWER_PACK_UNPACK_CODE = (
        'module attributes {transform.with_named_sequence} {\n'
        '  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {\n'
        '    %packs = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.pack">\n'
        '    %pad, %expand, %transpose = transform.structured.lower_pack %packs : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)\n'
        '    %unpacks = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.unpack">\n'
        '    %r0, %r1, %r2, %r3 = transform.structured.lower_unpack %unpacks : (!transform.op<"linalg.unpack">) -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">)\n'
        '    transform.yield\n'
        '  }\n'
        '}\n'
    )

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        packed_sizes = params["packed_sizes"]

        pack_transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %packed_op = transform.structured.pack %op packed_sizes = {packed_sizes} : (!transform.any_op) -> !transform.any_op\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f"    transform.annotate %packed_op \"tag\" = %tag : !transform.any_op, !transform.any_param\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )

        try:
            packed_code = run_transform_code(code, pack_transform_code)
            # Lower pack/unpack ops so the result is executable by the standard pipeline
            if "linalg.pack" in packed_code or "linalg.unpack" in packed_code:
                packed_code = run_transform_code(packed_code, cls.LOWER_PACK_UNPACK_CODE)
            return packed_code
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
