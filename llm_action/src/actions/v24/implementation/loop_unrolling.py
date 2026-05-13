from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopUnrolling(ActionBase):
    """
    Tile the innermost (reduction) loop dimension of the target operation
    then unroll the generated loop to expose instruction-level parallelism.

    The action tiles a single dimension and then unrolls the resulting loop.
    This is a two-step composite: tile to create a loop of known trip count,
    then unroll it to replicate the loop body.

    Structure-preserving (Category A): output still contains the linalg op.

    unique_execution = False because unrolling can target different dimensions
    at different levels after prior tiling.
    """

    unique_execution: bool = False  # can unroll different loops

    VOCAB = [2, 4, 8, 16, 32]  # unroll factor vocabulary

    @classmethod
    def parameters(cls) -> dict:
        return {
            "unroll_factor": {
                "description": "Number of loop body copies per iteration.",
                "type": "int",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        factor = params.get("unroll_factor")
        if factor is None or not isinstance(factor, int) or factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        factor = params["unroll_factor"]

        # Tile the last (reduction/innermost) dimension to create a loop,
        # then unroll that loop. We tile with [0, 0, factor] for 3-loop matmul.
        # For general ops, we tile only the last dimension.
        # We use a dynamic approach: tile the last dimension with the unroll factor,
        # then unroll the resulting loop.
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop0 = transform.structured.tile_using_for %op tile_sizes [0, 0, {factor}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.loop.unroll %loop0 {{factor = {factor}}} : !transform.any_op\n'
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
        idx = raw_slots[0] % len(cls.VOCAB)
        return {"unroll_factor": cls.VOCAB[idx]}
