import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class SplitReduction(ActionBase):
    """Split a reduction loop into independent partial reductions for parallelism.
    Decomposes the reduction into partial sums followed by a final combination."""

    unique_execution = True  # Structurally changes the reduction; second application targets the combine op

    VOCAB = [2, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {"split_factor": "number of independent partial reductions"}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        factor = params.get("split_factor", 0)
        if factor < 2:
            return False
        # Only ops with reduction dimensions: matmul, conv, pooling, generic with reductions
        has_reduction = (
            "linalg.matmul" in code
            or "linalg.conv_2d_nchw_fchw" in code
            or "linalg.pooling_nchw_max" in code
            or '"reduction"' in code
        )
        if not has_reduction:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        split_factor = params["split_factor"]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %init, %fill, %split, %combine = transform.structured.split_reduction %op {{split_factor = {split_factor} : i64, insert_split_dimension = 0 : i64}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %combine "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return [len(cls.VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {"split_factor": cls.VOCAB[raw_slots[0] % len(cls.VOCAB)]}
