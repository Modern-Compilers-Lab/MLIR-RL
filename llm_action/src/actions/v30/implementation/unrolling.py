from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """Replicate the loop body multiple times, reducing loop overhead and
    exposing independent operations for instruction-level parallelism.

    Tiles a selected dimension and fully unrolls the resulting inner loop.

    Repeatable: can unroll different dimensions in successive applications."""

    unique_execution: bool = True  # can unroll different loops in different steps

    VOCAB = [0, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Per-dimension unroll factor (0 = skip); first non-zero dim is tiled and its inner loop unrolled",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or all(s == 0 for s in tile_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]

        # Find first non-zero dimension to unroll
        unroll_dim = -1
        unroll_factor = 0
        for i, s in enumerate(tile_sizes):
            if s > 0:
                unroll_dim = i
                unroll_factor = s
                break

        if unroll_dim < 0:
            return code

        # Build tile_sizes with only the selected dimension non-zero
        effective_tile_sizes = [0] * len(tile_sizes)
        effective_tile_sizes[unroll_dim] = unroll_factor

        n_loops = 1  # only one non-zero tile size

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop = transform.structured.tile_using_for %op tile_sizes {effective_tile_sizes} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    %loop_cast = transform.cast %loop : !transform.any_op to !transform.op<"scf.for">\n'
            f'    transform.loop.unroll %loop_cast {{factor = {unroll_factor} : i64}} : !transform.op<"scf.for">\n'
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
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"tile_sizes": sizes}
