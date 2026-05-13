from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """Tile, bufferize, and promote operands into contiguous scratch buffers.

    Copies tiled operand slices into contiguous temporary buffers to eliminate
    non-unit-stride accesses and TLB pressure within the tile computation.
    Includes tiling and bufferization as integral steps, since promote requires
    memref (buffer) form.
    """

    # Bufferization is a one-shot lowering; cannot be applied twice.
    unique_execution: bool = True

    VOCAB = [16, 32, 64, 128]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for outer tiling before promotion.",
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
        if any(not isinstance(s, int) or s <= 0 for s in tile_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_loops = len(tile_sizes)
        loop_results = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n'
            f'\n'
            f'    // Step 1: Match and tile\n'
            f'    %op0 = transform.structured.match attributes{{tag = "operation_0"}} in %module : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op0 tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'\n'
            f'    // Tag the tiled op so we can find it after bufferization\n'
            f'    %tag_val = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag_val : !transform.any_op, !transform.any_param\n'
            f'\n'
            f'    // Step 2: Bufferize (consumes module handle, returns new one)\n'
            f'    %arg1 = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}} %module {{bufferize_function_boundaries = true}} : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 3: Re-match after bufferization\n'
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 4: Promote all operands\n'
            f'    %promoted_op = transform.structured.promote %promoted_target {{operands_to_promote = [0, 1, 2], use_alloca}} : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 5: Canonicalize (fold dynamic shapes to static)\n'
            f'    %func2 = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %func2 : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 6: Re-tag for downstream actions\n'
            f'    %final_op = transform.structured.match attributes{{tag = "tiled_target"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %final_op "tag" = %final_tag : !transform.any_op, !transform.any_param\n'
            f'\n'
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
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"tile_sizes": sizes}
