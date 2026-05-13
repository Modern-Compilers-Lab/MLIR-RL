from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """Copy tiled operand slices into contiguous temporary buffers.

    Repeatable: can promote at different tile levels after multi-level tiling.
    Requires tiling as internal preprocessing step, then bufferization, then promotion.
    """

    unique_execution: bool = False  # can promote at different tile scopes

    TILE_VOCAB = [0, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for the tiling step before promotion; 0 means do not tile.",
                "type": "list[int]",
                "values": cls.TILE_VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or not isinstance(tile_sizes, list):
            return False
        if all(s == 0 for s in tile_sizes):
            return False
        # Promotion requires linalg ops (not already lowered to loops)
        if "linalg." not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_loops = sum(1 for s in tile_sizes if s != 0)
        assert n_loops > 0

        loop_results = ", ".join(["!transform.any_op"] * n_loops)
        tile_sizes_str = str(tile_sizes)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %module : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {tile_sizes_str}'
            f' : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    %tiled_tag = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tiled_tag : !transform.any_op, !transform.any_param\n'
            f'    %bufferized = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}}'
            f' %module {{bufferize_function_boundaries = true}}'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}} in %bufferized : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted_op = transform.structured.promote %promoted_target'
            f' {{operands_to_promote = [0, 1, 2], use_alloca}}'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %promoted_op "tag" = %final_tag : !transform.any_op, !transform.any_param\n'
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
        return [len(cls.TILE_VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.TILE_VOCAB[raw_slots[i] % len(cls.TILE_VOCAB)] for i in range(n)]
        return {"tile_sizes": sizes}
