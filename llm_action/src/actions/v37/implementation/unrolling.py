import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """Tile a single loop dimension and unroll the resulting inner loop.

    Reduces loop overhead and exposes instruction-level parallelism.
    Repeated application on different dimensions is meaningful.
    """

    unique_execution: bool = True  # can unroll different dimensions

    LOOP_DIM_VOCAB = [0, 1, 2, 3]  # which loop dimension to target (capped at n_loops)
    UNROLL_FACTOR_VOCAB = [2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "loop_dim": {
                "description": "Which loop dimension to unroll (0-indexed).",
                "type": "int",
                "values": cls.LOOP_DIM_VOCAB,
            },
            "unroll_factor": {
                "description": "Unroll factor.",
                "type": "int",
                "values": cls.UNROLL_FACTOR_VOCAB,
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
        if unroll_factor not in cls.UNROLL_FACTOR_VOCAB:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        loop_dim = params["loop_dim"]
        unroll_factor = params["unroll_factor"]

        # Build tile_sizes: tile only the target dimension with unroll_factor
        # We need to determine the number of loops from the code by counting
        # iterator_types in the tagged operation
        import re
        m = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
        if not m:
            return code
        iters = [x.strip().strip('"') for x in m.group(1).split(",")]
        n_dims = len(iters)
        if loop_dim >= n_dims:
            return code

        tile_sizes = [0] * n_dims
        tile_sizes[loop_dim] = unroll_factor

        # Only 1 non-zero tile size = 1 loop
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.loop.unroll %loop {{factor = {unroll_factor} : i64}} : !transform.any_op\n'
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
        return 2

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        n_dim_choices = min(n_loops, len(cls.LOOP_DIM_VOCAB))
        return [n_dim_choices, len(cls.UNROLL_FACTOR_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n_dim_choices = min(n_loops, len(cls.LOOP_DIM_VOCAB))
        loop_dim = raw_slots[0] % n_dim_choices
        unroll_factor = cls.UNROLL_FACTOR_VOCAB[raw_slots[1] % len(cls.UNROLL_FACTOR_VOCAB)]
        return {"loop_dim": loop_dim, "unroll_factor": unroll_factor}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n_dim_choices = min(n_loops, len(cls.LOOP_DIM_VOCAB))
        # Slot 0: loop dim - all valid dims are ok
        dim_mask = np.ones(n_dim_choices, dtype=bool)
        # Slot 1: unroll factor must divide the loop bound of the chosen dim
        # Since we don't know which dim will be chosen, allow all factors
        factor_mask = np.ones(len(cls.UNROLL_FACTOR_VOCAB), dtype=bool)
        return np.concatenate([dim_mask, factor_mask])
