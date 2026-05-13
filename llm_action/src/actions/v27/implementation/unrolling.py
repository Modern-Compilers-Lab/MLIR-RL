from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """Tile and unroll the innermost loop to expose more independent operations.

    Tiles the target linalg operation and then unrolls the innermost generated
    loop by the specified factor.  This exposes multiple independent FMA
    operations to the out-of-order execution engine and enables register-level
    data reuse.  Includes tiling as an integral part so that the outermost
    generated loop can be tagged for downstream actions.
    """

    # Repeated application at different tile/unroll levels is a meaningful tuning knob.
    unique_execution: bool = True

    TILE_VOCAB = [4, 8, 16, 32]
    UNROLL_VOCAB = [2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes per loop dimension before unrolling.",
                "type": "list[int]",
                "values": cls.TILE_VOCAB,
            },
            "unroll_factor": {
                "description": "Number of times to replicate the innermost loop body.",
                "type": "int",
                "values": cls.UNROLL_VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        unroll_factor = params.get("unroll_factor", 0)
        if not tile_sizes or not isinstance(tile_sizes, list):
            return False
        if any(not isinstance(s, int) or s <= 0 for s in tile_sizes):
            return False
        if not isinstance(unroll_factor, int) or unroll_factor < 2:
            return False
        # The innermost tile size must be divisible by the unroll factor
        # to avoid generating remainder loops
        if tile_sizes[-1] % unroll_factor != 0:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        unroll_factor = params["unroll_factor"]
        n_loops = len(tile_sizes)
        loop_results = ", ".join(["!transform.any_op"] * n_loops)
        innermost_idx = n_loops - 1

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    transform.loop.unroll %loops#{innermost_idx} {{factor = {unroll_factor} : i64}} : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        if 'tag = "operation_0"' not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        n = min(n_loops, MAX_PARAM_SLOTS - 1)  # reserve 1 slot for unroll_factor
        slots = [len(cls.TILE_VOCAB)] * n + [len(cls.UNROLL_VOCAB)]
        return slots

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS - 1)
        tile_sizes = [cls.TILE_VOCAB[raw_slots[i] % len(cls.TILE_VOCAB)] for i in range(n)]
        unroll_factor = cls.UNROLL_VOCAB[raw_slots[n] % len(cls.UNROLL_VOCAB)]
        # Ensure innermost tile is divisible by unroll factor
        while tile_sizes[-1] % unroll_factor != 0:
            idx = cls.TILE_VOCAB.index(tile_sizes[-1])
            if idx < len(cls.TILE_VOCAB) - 1:
                tile_sizes[-1] = cls.TILE_VOCAB[idx + 1]
            else:
                # Fall back to smallest unroll factor
                unroll_factor = 2
                break
        return {"tile_sizes": tile_sizes, "unroll_factor": unroll_factor}
