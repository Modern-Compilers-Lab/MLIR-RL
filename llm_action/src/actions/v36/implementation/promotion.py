from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """Copy tiled operand slices into contiguous temporary buffers to eliminate
    non-unit-stride memory accesses. Requires the target op to already be tiled
    (inside scf.for loops) so that operands are subviews eligible for promotion.
    Structure-preserving: output is still a linalg op with promoted operands."""

    unique_execution: bool = True  # promoting different operand subsets is meaningful

    OPERAND_COMBOS = [
        [0, 1, 2],  # all operands
        [0, 1],     # both inputs
        [0],        # first input (filter)
        [1],        # second input (img2col)
        [2],        # output (accumulator)
    ]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "operands_to_promote": {
                "description": "List of operand indices to promote into contiguous local buffers.",
                "type": "list[int]",
                "values": cls.OPERAND_COMBOS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        operands = params.get("operands_to_promote", [])
        if not operands or not isinstance(operands, list):
            return False
        if any(not isinstance(o, int) or o < 0 for o in operands):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        operands = params["operands_to_promote"]
        operands_str = ", ".join(str(o) for o in operands)

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.consumed}}) {{\n"
            # Bufferize the whole module (invalidates all prior handles)
            f"    %buf = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}} %arg1 {{bufferize_function_boundaries = true}} : (!transform.any_op) -> !transform.any_op\n"
            # Re-match the target op after bufferization
            f'    %func1 = transform.structured.match attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    %target = transform.structured.match attributes{{tag = "operation_0"}} in %func1 : (!transform.any_op) -> !transform.any_op\n'
            # Promote
            f"    %promoted = transform.structured.promote %target {{operands_to_promote = [{operands_str}], use_alloca}} : (!transform.any_op) -> !transform.any_op\n"
            # Canonicalize to fold dynamic shapes to static
            f'    %func2 = transform.structured.match attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %func2 : (!transform.any_op) -> !transform.any_op\n'
            # Re-tag for downstream actions
            f'    %func3 = transform.structured.match attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_op = transform.structured.match attributes{{tag = "operation_0"}} in %func3 : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f"    transform.annotate %final_op \"tag\" = %final_tag : !transform.any_op, !transform.any_param\n"
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
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.OPERAND_COMBOS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        idx = raw_slots[0] % len(cls.OPERAND_COMBOS)
        return {"operands_to_promote": cls.OPERAND_COMBOS[idx]}
