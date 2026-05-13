from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class SplitReduction(ActionBase):
    """
    Split a reduction loop into multiple partial reductions that execute
    independently, followed by a final combination step. Enables parallel
    accumulation across threads when outer parallel dimensions are insufficient.
    """

    unique_execution: bool = True  # structural replacement; output shape changes

    SPLIT_FACTORS = [2, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "split_factor": {
                "description": "Number of independent partial reductions to create.",
                "type": "int",
                "values": cls.SPLIT_FACTORS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        factor = params.get("split_factor", 0)
        if factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        factor = params["split_factor"]

        # split_reduction returns 4 handles: init_or_alloc, fill, split_linalg, combining
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %init_or_alloc, %fill, %split_linalg, %combining = transform.structured.split_reduction %op'
            f' {{split_factor = {factor}, insert_split_dimension = 0}}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %split_linalg "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.SPLIT_FACTORS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        idx = raw_slots[0] % len(cls.SPLIT_FACTORS)
        return {"split_factor": cls.SPLIT_FACTORS[idx]}
