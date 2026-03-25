from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopUnrolling(ActionBase):
    """Unrolls a loop dimension of a tagged linalg operation.

    Tiles the operation on a single dimension by the unroll factor, then
    unrolls the resulting loop to reduce loop control overhead and expose
    more independent operations for out-of-order scheduling.
    """

    UNROLL_VOCAB = [2, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "dimension": {
                "description": "Which loop dimension to unroll (0-indexed).",
                "type": "int",
            },
            "unroll_factor": {
                "description": "How many times to replicate the loop body.",
                "type": "int",
                "values": cls.UNROLL_VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        dimension = params.get("dimension")
        unroll_factor = params.get("unroll_factor")
        if not isinstance(dimension, int) or dimension < 0:
            return False
        if not isinstance(unroll_factor, int) or unroll_factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        dimension = params["dimension"]
        unroll_factor = params["unroll_factor"]

        tile_sizes = [0] * (dimension + 1)
        tile_sizes[dimension] = unroll_factor

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loop = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f"    transform.annotate %tiled_op \"tag\" = %tag : !transform.any_op, !transform.any_param\n"
            f"    transform.loop.unroll %loop {{factor = {unroll_factor}}} : !transform.any_op\n"
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
        return 2

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [min(n_loops, MAX_PARAM_SLOTS), len(cls.UNROLL_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n_dims = min(n_loops, MAX_PARAM_SLOTS)
        dimension = raw_slots[0] % n_dims
        unroll_factor = cls.UNROLL_VOCAB[raw_slots[1] % len(cls.UNROLL_VOCAB)]
        return {"dimension": dimension, "unroll_factor": unroll_factor}
