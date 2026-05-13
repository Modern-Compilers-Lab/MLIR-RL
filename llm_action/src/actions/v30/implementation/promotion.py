import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """Copy tiled operand slices into contiguous temporary buffers,
    eliminating irregular stride patterns from tiling subviews.

    Repeatable: can promote different operand subsets at different
    tiling scopes."""

    unique_execution: bool = True  # can promote different operands at different scopes

    # Operand promotion selections
    OPERAND_OPTIONS = [
        [0],          # promote input only
        [1],          # promote filter only
        [2],          # promote output only
        [0, 1],       # promote input + filter
        [0, 1, 2],    # promote all operands
    ]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "operands_to_promote": {
                "description": "Which operand indices to copy into contiguous buffers",
                "type": "list[int]",
                "values": ["[0]", "[1]", "[2]", "[0,1]", "[0,1,2]"],
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
    def _count_op_dims(cls, code: str) -> int:
        """Count loop dimensions of the tagged operation from the code."""
        if "linalg.conv_2d_nchw_fchw" in code:
            return 7
        match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
        if match:
            return len(match.group(1).split(","))
        return 4  # safe fallback for post-img2col generic

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        operands = params["operands_to_promote"]
        operands_str = str(operands).replace(" ", "")

        # Determine loop count to generate correct tile_sizes
        n_dims = cls._count_op_dims(code)
        tile_sizes = [4, 4] + [0] * (n_dims - 2)

        # Internal tiling + bufferization + promotion + canonicalization
        # Tile with moderate sizes to create subviews, then bufferize and promote
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n'
            f'\n'
            f'    // Step 1: Match and tile to create subviews for promotion\n'
            f'    %op0 = transform.structured.match attributes{{tag = "operation_0"}} in %module : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:2 = transform.structured.tile_using_for %op0 tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n'
            f'\n'
            f'    // Tag the tiled op so we can find it after bufferization\n'
            f'    %tag_val = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag_val : !transform.any_op, !transform.any_param\n'
            f'\n'
            f'    // Step 2: Bufferize (invalidates ALL handles)\n'
            f'    %buf = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}} %module {{bufferize_function_boundaries = true}} : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 3: Re-match after bufferization\n'
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 4: Promote with stack allocation (use_alloca avoids buffer-deallocation issues)\n'
            f'    %promoted_op = transform.structured.promote %promoted_target {{operands_to_promote = {operands_str}, use_alloca}} : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 5: Canonicalize (fold dynamic shapes to static)\n'
            f'    %main_fn = transform.structured.match attributes{{llvm.emit_c_interface}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %main_fn : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 6: Re-tag for downstream actions\n'
            f'    %final_op = transform.structured.match attributes{{tag = "tiled_target"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
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
        return [len(cls.OPERAND_OPTIONS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        idx = raw_slots[0] % len(cls.OPERAND_OPTIONS)
        return {"operands_to_promote": cls.OPERAND_OPTIONS[idx]}
