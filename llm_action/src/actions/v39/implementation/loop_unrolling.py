from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopUnrolling(ActionBase):
    """Replicate the body of the innermost loop multiple times to reduce loop overhead
    and expose ILP. Repeatable: can unroll different loops at different levels."""

    unique_execution: bool = True  # can unroll distinct loops

    FACTOR_VOCAB = [2, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "unroll_factor": {
                "description": "How many times to replicate the innermost loop body.",
                "type": "int",
                "values": cls.FACTOR_VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        factor = params.get("unroll_factor")
        if not factor or not isinstance(factor, int) or factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        factor = params["unroll_factor"]

        # Strategy: tile the innermost dimension by the unroll factor,
        # then unroll the resulting inner loop.
        # We tile with [0,...,0, factor] to only tile the last dimension,
        # producing one loop that we can unroll.
        # But we need to know n_loops... we'll use a different approach:
        # Match the tagged op, tile its last dim by factor, tag the tiled op,
        # then unroll the generated inner loop.

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop = transform.structured.tile_using_for %op tile_sizes [{factor}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    // Tag tiled op before unrolling (unroll invalidates nested handles)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    // Unroll the generated loop\n'
            f'    transform.loop.unroll %loop {{factor = {factor}}} : !transform.any_op\n'
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
        return [len(cls.FACTOR_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        idx = raw_slots[0] % len(cls.FACTOR_VOCAB)
        return {"unroll_factor": cls.FACTOR_VOCAB[idx]}
