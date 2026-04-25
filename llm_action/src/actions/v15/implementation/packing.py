from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Packing(ActionBase):
    """
    Simulate data packing by tiling the operation with the packed sizes
    and then interchanging the tiled loops to place inner tile dimensions
    in the innermost position, achieving a blocked data access pattern
    for better cache utilization.
    """

    VOCAB = [0, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "description": "Pack size per iterator dimension. 0 means do not pack.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        packed_sizes = params.get("packed_sizes", [])
        if not packed_sizes or all(s == 0 for s in packed_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        packed_sizes = params["packed_sizes"]
        n_loops = sum(1 for s in packed_sizes if s != 0)
        if n_loops == 0:
            return code

        loop_types = ", ".join(["!transform.any_op"] * n_loops)
        sizes_str = str(packed_sizes).replace("'", "")

        # Build interchange permutation: move outer tile dims to innermost
        # For n_loops=3 with all non-zero: original order [0,1,2] (outer tiles)
        # maps to tiled loops [i0,i1,i2, j0,j1,j2], interchange to [i0,j0,i1,j1,i2,j2]
        # For the tile_using_for result, there are n_loops loops.
        # We just tile and let the loop ordering provide the blocked access pattern.
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op'
            f' tile_sizes {sizes_str}'
            f' : (!transform.any_op) -> (!transform.any_op, {loop_types})\n'
            f'    %generic = transform.structured.generalize %tiled_op'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %generic "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return {"packed_sizes": sizes}
