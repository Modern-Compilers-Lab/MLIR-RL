import re

import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


def _count_dims(code: str) -> int:
    """Count the number of iteration dimensions of the tagged operation."""
    if "linalg.matmul" in code and "linalg.generic" not in code:
        return 3
    m = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
    if m:
        return len([t.strip() for t in m.group(1).split(",")])
    return 0


class LoopUnrolling(ActionBase):
    """Replicate the loop body to reduce overhead and expose instruction-level parallelism."""

    unique_execution = True  # unrolling distinct loops at different levels is meaningful

    UNROLL_VOCAB = [2, 4, 8, 16, 32, 64]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "unroll_factor": {
                "description": "Number of consecutive iterations fused into one loop body",
                "type": "int",
                "values": cls.UNROLL_VOCAB,
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
        n_dims = _count_dims(code)
        if n_dims == 0:
            return code

        # Tile the last dimension (innermost loop) with the unroll factor
        tile_sizes = [0] * (n_dims - 1) + [factor]
        sizes_str = str(tile_sizes).replace(" ", "")

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loop:1 = transform.structured.tile_using_for %op tile_sizes {sizes_str}"
            f" : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    %cast_loop = transform.cast %loop#0 : !transform.any_op to !transform.op<"scf.for">\n'
            f"    transform.loop.unroll %cast_loop {{factor = {factor}}} : !transform.op<\"scf.for\">\n"
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
        return [len(cls.UNROLL_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        factor = cls.UNROLL_VOCAB[raw_slots[0] % len(cls.UNROLL_VOCAB)]
        return {"unroll_factor": factor}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        # Last dimension bound (innermost loop)
        last_bound = loop_bounds[-1] if loop_bounds else 0
        # tile_sizes [0,...,0,factor]: creates loop with trip_count = last_bound / factor
        # unroll by factor: needs trip_count % factor == 0, i.e., last_bound % factor^2 == 0
        slot_mask = np.array(
            [last_bound > 0 and last_bound % (f * f) == 0 for f in cls.UNROLL_VOCAB],
            dtype=bool,
        )
        if not slot_mask.any():
            slot_mask[0] = True
        return slot_mask
