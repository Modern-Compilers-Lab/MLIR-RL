import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """Unroll loop iterations to reduce branch overhead and expose ILP.

    Pooling reduction windows are often very small (1x1 to 7x7), so loop overhead
    dominates. This action tiles the innermost loop dimension by factor 1 to isolate
    it as an scf.for, then unrolls that loop by the given factor.

    The action targets a specific loop dimension (selected by loop_dim) and unrolls
    it by the specified factor. It works by tiling that dimension by 1 to expose
    an scf.for loop, then applying transform.loop.unroll.

    Repeatable: unrolling distinct loops (different loop_dim) is meaningful.
    """

    unique_execution: bool = True  # unrolling different dimensions is a valid tuning knob

    FACTOR_VOCAB = [2, 4, 8, 16]  # unroll factors
    DIM_VOCAB = [0, 1, 2, 3, 4]  # which loop dimension to target (max 5 for pooling's 6 dims, skipping batch)

    @classmethod
    def parameters(cls) -> dict:
        return {
            "loop_dim": {
                "description": "Which loop dimension to unroll (0-indexed).",
                "type": "int",
                "values": cls.DIM_VOCAB,
            },
            "unroll_factor": {
                "description": "Unroll factor for the selected loop.",
                "type": "int",
                "values": cls.FACTOR_VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        loop_dim = params.get("loop_dim")
        factor = params.get("unroll_factor")
        if loop_dim is None or factor is None:
            return False
        if factor < 2:
            return False
        if loop_dim < 0:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        loop_dim = params["loop_dim"]
        factor = params["unroll_factor"]

        # Build tile_sizes: all zeros except the target dimension which is 1
        # We need to know how many dimensions the op has. For pooling_nchw_max it's 6.
        # We'll use a large enough list and pad with zeros.
        max_dims = loop_dim + 1
        tile_sizes = [0] * max_dims
        tile_sizes[loop_dim] = 1

        # Tiling dim with size 1 produces exactly 1 loop
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop = transform.structured.tile_using_for %op tile_sizes {str(tile_sizes)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return 2  # loop_dim + unroll_factor

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        n_dim_choices = min(n_loops, len(cls.DIM_VOCAB))
        return [n_dim_choices, len(cls.FACTOR_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n_dim_choices = min(n_loops, len(cls.DIM_VOCAB))
        dim_idx = raw_slots[0] % n_dim_choices
        loop_dim = cls.DIM_VOCAB[dim_idx] if dim_idx < len(cls.DIM_VOCAB) else dim_idx
        factor_idx = raw_slots[1] % len(cls.FACTOR_VOCAB)
        return {
            "loop_dim": loop_dim,
            "unroll_factor": cls.FACTOR_VOCAB[factor_idx],
        }
