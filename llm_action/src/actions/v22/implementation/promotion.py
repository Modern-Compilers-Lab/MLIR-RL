from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """
    Copy tiled operand data into contiguous local buffers to eliminate irregular
    strides and TLB pressure within tiled blocks.

    Requires tiling + bufferization as preprocessing. Promoting different operand
    subsets is a meaningful tuning knob, so repeated application is allowed.
    """

    unique_execution: bool = False  # promoting different operand sets is meaningful

    OPERAND_SETS = [
        [0, 1, 2],  # all operands
        [0, 1],     # inputs only
        [0, 2],     # LHS + output
        [1, 2],     # RHS + output
        [1],        # RHS only (column-major stride fix)
    ]

    # Fixed tile sizes for the internal tiling preprocessing step
    TILE_SIZES = [32, 32, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "operands_to_promote": {
                "description": "Which operand indices to promote into contiguous local buffers.",
                "type": "list[int]",
                "values": cls.OPERAND_SETS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        operands = params.get("operands_to_promote", [])
        if not operands:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        operands = params["operands_to_promote"]
        operands_str = "[" + ", ".join(str(o) for o in operands) + "]"
        tile_sizes_str = str(cls.TILE_SIZES)

        n_loops = sum(1 for s in cls.TILE_SIZES if s != 0)
        loop_handles = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n'
            f'\n'
            f'    // Step 1: Match and tile\n'
            f'    %op0 = transform.structured.match attributes{{tag = "operation_0"}} in %module : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op0 tile_sizes {tile_sizes_str} : (!transform.any_op) -> (!transform.any_op, {loop_handles})\n'
            f'\n'
            f'    // Tag the tiled op so we can find it after bufferization\n'
            f'    %tag_val = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag_val : !transform.any_op, !transform.any_param\n'
            f'\n'
            f'    // Step 2: Bufferize the module (invalidates ALL handles)\n'
            f'    %buf_module = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}} %module {{bufferize_function_boundaries = true}} : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 3: Re-match after bufferization\n'
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}} in %buf_module : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 4: Promote (use_alloca avoids memref.dealloc conflicts with buffer-deallocation-pipeline)\n'
            f'    %promoted_op = transform.structured.promote %promoted_target {{operands_to_promote = {operands_str}, use_alloca}} : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 5: Canonicalize (fold dynamic shapes to static)\n'
            f'    %canon_module = transform.apply_registered_pass "canonicalize" to %buf_module : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 6: Re-tag for downstream actions\n'
            f'    %final_op = transform.structured.match attributes{{tag = "tiled_target"}} in %canon_module : (!transform.any_op) -> !transform.any_op\n'
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
        return True

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.OPERAND_SETS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        idx = raw_slots[0] % len(cls.OPERAND_SETS)
        return {"operands_to_promote": cls.OPERAND_SETS[idx]}
