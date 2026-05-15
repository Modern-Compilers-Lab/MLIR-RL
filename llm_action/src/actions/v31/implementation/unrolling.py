from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """Unroll a loop to eliminate loop overhead and increase ILP.

    Tiles one loop dimension and then unrolls the resulting outer loop.
    The first non-zero entry in tile_sizes selects the target loop dimension
    and the tile/unroll factor. Particularly useful for small reduction loops
    (KH, KW) in pooling.
    """

    # Different loops can be unrolled independently.
    unique_execution: bool = True

    VOCAB = [0, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "First non-zero entry selects the loop dimension and unroll factor. "
                "All other entries must be 0.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or not isinstance(tile_sizes, list):
            return False
        non_zero = [s for s in tile_sizes if s != 0]
        if len(non_zero) != 1:
            return False
        if non_zero[0] < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]

        # Find the single non-zero dimension
        unroll_factor = 0
        for s in tile_sizes:
            if s != 0:
                unroll_factor = s
                break

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop0 = transform.structured.tile_using_for %op tile_sizes {tile_sizes}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.loop.unroll %loop0 {{factor = {unroll_factor}}} : !transform.any_op\n'
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
        # Ensure only one non-zero entry: pick the first non-zero
        found = False
        result = []
        for s in sizes:
            if s != 0 and not found:
                result.append(s)
                found = True
            else:
                result.append(0)
        return {"tile_sizes": result}
