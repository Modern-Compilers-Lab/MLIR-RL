from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """Unroll inner loops to expose more independent operations and reduce loop overhead.
    Tiles one dimension then unrolls the resulting scf.for loop.
    Structure-preserving at the linalg level: the tiled linalg op remains inside the unrolled loop body."""

    unique_execution: bool = True  # unrolling distinct loop dimensions is meaningful

    UNROLL_VOCAB = [2, 4, 8, 16]  # unroll factors

    @classmethod
    def parameters(cls) -> dict:
        return {
            "loop_dim": {
                "description": "Which loop dimension to unroll (0-indexed).",
                "type": "int",
            },
            "unroll_factor": {
                "description": "Unroll factor.",
                "type": "int",
                "values": cls.UNROLL_VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        loop_dim = params.get("loop_dim")
        unroll_factor = params.get("unroll_factor")
        if loop_dim is None or unroll_factor is None:
            return False
        if not isinstance(loop_dim, int) or loop_dim < 0:
            return False
        if not isinstance(unroll_factor, int) or unroll_factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        loop_dim = params["loop_dim"]
        unroll_factor = params["unroll_factor"]

        # Build tile_sizes: unroll_factor at the target dimension, 0 elsewhere
        # We need to know the number of dimensions, but we don't parse IR.
        # Use MAX_PARAM_SLOTS as the max and pad with zeros.
        # tile_using_for ignores trailing zeros for dimensions that don't exist.
        tile_sizes = [0] * (loop_dim + 1)
        tile_sizes[loop_dim] = unroll_factor

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loop = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, !transform.op<\"scf.for\">)\n"
            # Tag the tiled op BEFORE unrolling (unrolling invalidates nested handles)
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f"    transform.annotate %tiled_op \"tag\" = %tag : !transform.any_op, !transform.any_param\n"
            f"    transform.loop.unroll %loop {{factor = {unroll_factor}}} : !transform.op<\"scf.for\">\n"
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
        n_dim_choices = min(n_loops, MAX_VOCAB_SIZE_PER_SLOT)
        if n_dim_choices < 1:
            n_dim_choices = 1
        return [n_dim_choices, len(cls.UNROLL_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n_dim_choices = min(n_loops, MAX_VOCAB_SIZE_PER_SLOT)
        if n_dim_choices < 1:
            n_dim_choices = 1
        loop_dim = raw_slots[0] % n_dim_choices
        unroll_factor = cls.UNROLL_VOCAB[raw_slots[1] % len(cls.UNROLL_VOCAB)]
        return {"loop_dim": loop_dim, "unroll_factor": unroll_factor}
