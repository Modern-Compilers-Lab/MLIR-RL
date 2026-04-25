import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Padding(ActionBase):
    """Pad operand dimensions to multiples of a given value using transform.structured.pad."""

    VOCAB = [2, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "pad_multiple": {
                "description": "Pad dimensions to be multiples of this value.",
                "type": "int",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def _detect_element_type(cls, code: str) -> str:
        if "xf32>" in code or "xf32," in code:
            return "f32"
        return "f64"

    @classmethod
    def _count_operands(cls, code: str) -> int:
        if "linalg.matmul" in code:
            return 3
        if "linalg.conv_2d" in code:
            return 3
        match = re.search(r"ins\(([^)]+)\)\s*outs\(([^)]+)\)", code)
        if match:
            n_ins = len([x for x in match.group(1).split(",") if "%" in x])
            n_outs = len([x for x in match.group(2).split(",") if "%" in x])
            return n_ins + n_outs
        return 3

    @classmethod
    def _count_dims(cls, code: str) -> int:
        if "linalg.matmul" in code:
            return 3
        if "linalg.generic" in code:
            n = code.count('"parallel"') + code.count('"reduction"')
            return n if n > 0 else 3
        return 3

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        pad_multiple = params.get("pad_multiple", 0)
        if pad_multiple <= 0:
            return False
        if "linalg." not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        code = cls.preprocess(code, params)
        pad_multiple = params["pad_multiple"]
        elem_type = cls._detect_element_type(code)
        n_operands = cls._count_operands(code)
        n_dims = cls._count_dims(code)

        padding_values = ", ".join([f"0.0 : {elem_type}"] * n_operands)
        padding_dimensions = str(list(range(n_dims)))
        pad_multiples = str([pad_multiple] * n_dims)

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %padded, %pad, %copy = transform.structured.pad %op pad_to_multiple_of {pad_multiples} {{\n"
            f"      padding_values = [{padding_values}],\n"
            f"      padding_dimensions = {padding_dimensions},\n"
            f'      copy_back_op = "linalg.copy"\n'
            f"    }} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f"    transform.annotate %padded \"tag\" = %tag : !transform.any_op, !transform.any_param\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )

        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if before.strip() == after.strip():
            return False
        if "func.func" not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {"pad_multiple": cls.VOCAB[raw_slots[0] % len(cls.VOCAB)]}
