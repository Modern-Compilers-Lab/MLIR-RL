from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Packing(ActionBase):
    """Transform tensor data layout into a blocked/tiled organization via linalg.pack.

    Repeatable: can pack different dimensions at different levels.
    """

    unique_execution: bool = False  # can pack different dimensions separately

    VOCAB = [0, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "description": "Pack size per iterator dimension; 0 means do not pack that dimension.",
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
        if "linalg." not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        packed_sizes = params["packed_sizes"]
        packed_sizes_str = "[" + ", ".join(str(s) for s in packed_sizes) + "]"

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %packed = transform.structured.pack %op packed_sizes = {packed_sizes_str}'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %packed "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'\n'
            f'    %pack_ops = transform.structured.match ops{{["linalg.pack"]}} in %arg1'
            f' : (!transform.any_op) -> !transform.op<"linalg.pack">\n'
            f'    %pad, %expand, %transpose = transform.structured.lower_pack %pack_ops'
            f' : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)\n'
            f'\n'
            f'    %func1 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "linalg-generalize-named-ops" to %func1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %func2 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %func2 : (!transform.any_op) -> !transform.any_op\n'
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
