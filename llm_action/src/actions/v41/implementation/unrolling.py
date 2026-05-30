import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """Replicate loop body to reduce branch overhead and expose ILP.
    Tiles a target dimension then unrolls the resulting loop.
    """

    unique_execution = True

    FACTOR_VOCAB = [2, 4, 8, 16, 32, 64]  # unroll factors

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "type": "list[int]",
                "description": "Tile sizes (unroll factor at target dim, 0 elsewhere)",
            },
            "unroll_factor": {
                "type": "int",
                "description": "Unroll factor for the target loop",
                "values": cls.FACTOR_VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        factor = params.get("unroll_factor", 0)
        if not tile_sizes or factor <= 1:
            return False
        if all(s == 0 for s in tile_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        factor = params["unroll_factor"]

        # tile_sizes has exactly 1 non-zero entry -> produces exactly 1 loop handle
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loop = transform.structured.tile_using_for %op tile_sizes {tile_sizes}"
            f" : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f"    transform.loop.unroll %loop {{factor = {factor}}} : !transform.any_op\n"
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
        return [n_dim_choices, len(cls.FACTOR_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n_dim = min(n_loops, MAX_VOCAB_SIZE_PER_SLOT)
        loop_dim = raw_slots[0] % n_dim
        factor = cls.FACTOR_VOCAB[raw_slots[1] % len(cls.FACTOR_VOCAB)]
        tile_sizes = [0] * n_loops
        tile_sizes[loop_dim] = factor
        return {"tile_sizes": tile_sizes, "unroll_factor": factor}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n_dim = min(n_loops, MAX_VOCAB_SIZE_PER_SLOT)
        # Slot 0 (loop dim): all dims are valid targets
        dim_mask = np.ones(n_dim, dtype=bool)
        # Slot 1 (factor): factor must divide ALL loop bounds (conservative)
        factor_mask = np.array(
            [all(b > 0 and b % f == 0 for b in loop_bounds[:n_loops]) for f in cls.FACTOR_VOCAB],
            dtype=bool,
        )
        if not factor_mask.any():
            factor_mask[0] = True
        return np.concatenate([dim_mask, factor_mask])
