from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """Tile a single loop dimension and unroll it to increase ILP.

    Repeatable: can unroll different loop dimensions across applications.
    """

    unique_execution: bool = True  # can unroll different dimensions in successive steps

    DIM_VOCAB_SIZE = 4  # support up to 4 loop dims (enough for 4D add)
    FACTOR_VOCAB = [2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "loop_dim": "Which loop dimension to unroll (0-indexed).",
            "unroll_factor": "Unroll factor. Values: [2, 4, 8, 16].",
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        loop_dim = params.get("loop_dim", -1)
        unroll_factor = params.get("unroll_factor", 0)
        if loop_dim < 0 or unroll_factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        loop_dim = params["loop_dim"]
        unroll_factor = params["unroll_factor"]

        # Build tile_sizes: unroll_factor at target dim, 0 elsewhere
        tile_sizes = [0] * (loop_dim + 1)
        tile_sizes[loop_dim] = unroll_factor

        # Tile to create one scf.for loop, tag the tiled op BEFORE unrolling
        # (unrolling invalidates handles to nested ops)
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loop:1 = transform.structured.tile_using_for %op tile_sizes {str(tile_sizes)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f"    transform.annotate %tiled_op \"tag\" = %tag : !transform.any_op, !transform.any_param\n"
            f"    transform.loop.unroll %loop#0 {{factor = {unroll_factor}}} : !transform.any_op\n"
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
        return [max(min(n_loops, cls.DIM_VOCAB_SIZE), 1), len(cls.FACTOR_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n_dims = max(min(n_loops, cls.DIM_VOCAB_SIZE), 1)
        loop_dim = raw_slots[0] % n_dims
        unroll_factor = cls.FACTOR_VOCAB[raw_slots[1] % len(cls.FACTOR_VOCAB)]
        return {"loop_dim": loop_dim, "unroll_factor": unroll_factor}
