import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopUnrolling(ActionBase):
    """Replicate the loop body a fixed number of times per iteration.
    Implemented as tile(factor) on the first dimension + unroll the resulting loop.

    unique_execution = False: can unroll different loops at different levels."""

    unique_execution: bool = False  # can unroll different loops after tiling

    VOCAB = [2, 4, 8, 16]  # unroll factors

    @classmethod
    def parameters(cls) -> dict:
        return {
            "unroll_factor": {
                "description": "Number of loop body copies per iteration",
                "type": "int",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        factor = params.get("unroll_factor", 0)
        if factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        factor = params["unroll_factor"]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop = transform.structured.tile_using_for %op'
            f' tile_sizes [{factor}]'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.loop.unroll %loop {{factor = {factor}}} : !transform.op<"scf.for">\n'
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
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        idx = raw_slots[0] % len(cls.VOCAB)
        return {"unroll_factor": cls.VOCAB[idx]}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds or not loop_bounds[0]:
            return None
        bound = loop_bounds[0]
        # factor^2 must divide the first dimension bound (tile creates step=factor, unroll replicates factor times)
        slot_mask = np.array([
            bound > 0 and bound % (f * f) == 0
            for f in cls.VOCAB
        ], dtype=bool)
        if not slot_mask.any():
            slot_mask[0] = True
        return slot_mask
