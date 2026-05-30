import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopUnrolling(ActionBase):
    """Unroll the innermost loop of a tiled operation to increase ILP.

    Operates on a tagged scf.for loop (after tiling) — unrolls the innermost loop
    body multiple times per iteration. The action tiles the tagged operation first
    to create the loop structure, then unrolls the innermost loop.

    Structure-preserving for loops — repeatable for different loops/factors.
    """

    # Can unroll different loops at different factors after re-tiling — repeatable.
    unique_execution: bool = True

    UNROLL_VOCAB = [2, 4, 8, 16, 32]  # unroll factors

    @classmethod
    def parameters(cls) -> dict:
        return {
            "unroll_factor": {
                "description": "Number of times to replicate the loop body.",
                "type": "int",
                "values": cls.UNROLL_VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        factor = params.get("unroll_factor", 0)
        if not isinstance(factor, int) or factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        factor = params["unroll_factor"]

        # Match the tagged op. If it's a linalg op, tile it first to create loops,
        # then unroll the innermost. If it's already an scf.for (after tiling),
        # unroll it directly.
        # We use a two-strategy approach:
        # Strategy 1: tagged op is scf.for — unroll directly
        # Strategy 2: tagged op is linalg — tile with factor, then unroll

        # Try strategy: match as any op, tile with the unroll factor on the
        # innermost (reduction) dimension, then unroll that loop.
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    // Tile the innermost dimension with unroll_factor, creating one loop\n'
            f'    %tiled_op, %inner_loop = transform.structured.tile_using_for %op'
            f' tile_sizes [{factor}]'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    // Tag the tiled op before unrolling (unroll invalidates nested handles)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    // Unroll the loop\n'
            f'    %cast_loop = transform.cast %inner_loop : !transform.any_op to !transform.op<"scf.for">\n'
            f'    transform.loop.unroll %cast_loop {{factor = {factor}}} : !transform.op<"scf.for">\n'
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
        return [len(cls.UNROLL_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        idx = raw_slots[0] % len(cls.UNROLL_VOCAB)
        return {"unroll_factor": cls.UNROLL_VOCAB[idx]}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        # The unroll factor tiles the first dimension, so it must divide the first loop bound
        bound = loop_bounds[0] if loop_bounds else 0
        mask = np.array([
            bound > 0 and bound % f == 0
            for f in cls.UNROLL_VOCAB
        ], dtype=bool)
        if not mask.any():
            mask[0] = True
        return mask
