from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopDistribution(ActionBase):
    """Splits a reduction dimension into partial reductions using split_reduction.

    Decomposes a single reduction into multiple independent partial reductions
    that can be optimized separately (e.g. vectorized, parallelized).  The
    ``transform.structured.split_reduction`` op creates an init/fill, a partial
    reduction (splitting) loop, and a combining reduction.  The combining
    operation is re-annotated with ``tag = "operation_0"`` so that subsequent
    transforms can continue to target the main operation.
    """

    SPLIT_OPTIONS = [2, 4, 8, 16, 32]

    # ------------------------------------------------------------------
    # ActionBase interface
    # ------------------------------------------------------------------

    @classmethod
    def parameters(cls) -> dict:
        return {
            "split_factor": {
                "description": "Number of partial reduction chunks to split the reduction into.",
                "type": "int",
                "values": cls.SPLIT_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        split_factor = params.get("split_factor")
        if not isinstance(split_factor, int) or split_factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        split_factor = params["split_factor"]

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %init, %splitting, %combining, %residual = transform.structured.split_reduction %op {{ split_factor = {split_factor}, insert_split_dimension = 0 }} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %combining "tag" = %tag : !transform.any_op, !transform.any_param\n'
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

    # ------------------------------------------------------------------
    # RL parameter interface
    # ------------------------------------------------------------------

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.SPLIT_OPTIONS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        split_factor = cls.SPLIT_OPTIONS[raw_slots[0] % len(cls.SPLIT_OPTIONS)]
        return {"split_factor": split_factor}
