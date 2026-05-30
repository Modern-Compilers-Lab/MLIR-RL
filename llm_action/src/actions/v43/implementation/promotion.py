import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """Copy tiled operand slices into contiguous local buffers to eliminate strided access.

    Internally tiles with fixed sizes [16,16,0,0,16,0,0], bufferizes, then promotes
    selected operands. Uses use_alloca to avoid buffer-deallocation issues.

    Repeatable: can promote different operand subsets in successive applications.
    """

    unique_execution = False  # can promote different operand subsets

    # Predefined operand combinations for promotion
    OPERAND_COMBOS = [
        [0],          # input only
        [1],          # filter only
        [2],          # output only
        [0, 1],       # input + filter
        [1, 2],       # filter + output
        [0, 1, 2],    # all operands
    ]

    # Fixed internal tile sizes: 16 divides all N, F, C values in the conv2d dataset
    INTERNAL_TILE_SIZES = [16, 16, 0, 0, 16, 0, 0]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "operands_to_promote": {
                "description": "List of operand indices to promote into contiguous local buffers",
                "type": "list[int]",
                "values": cls.OPERAND_COMBOS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        operands = params.get("operands_to_promote", [])
        if not operands:
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
        operands = params["operands_to_promote"]
        tile_sizes = cls.INTERNAL_TILE_SIZES
        n_loops = sum(1 for s in tile_sizes if s != 0)
        loop_types = ", ".join(["!transform.any_op"] * n_loops)
        operands_str = ", ".join(str(o) for o in operands)

        # Complete promotion sequence:
        # 1. Match and tile
        # 2. Tag tiled op for re-matching after bufferization
        # 3. Bufferize (consumes module, invalidates all handles)
        # 4. Re-match tiled op by tag
        # 5. Promote
        # 6. Canonicalize
        # 7. Re-tag for downstream
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n"
            # Step 1: Match and tile
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %module : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {str(tile_sizes)}"
            f" : (!transform.any_op) -> (!transform.any_op, {loop_types})\n"
            # Step 2: Tag tiled op
            f'    %tag_val = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag_val : !transform.any_op, !transform.any_param\n'
            # Step 3: Bufferize
            f"    %buf = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}} %module"
            f" {{bufferize_function_boundaries = true}} : (!transform.any_op) -> !transform.any_op\n"
            # Step 4: Re-match after bufferization
            f'    %func1 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}} in %func1 : (!transform.any_op) -> !transform.any_op\n'
            # Step 5: Promote
            f"    %promoted_op = transform.structured.promote %promoted_target"
            f" {{operands_to_promote = [{operands_str}], use_alloca}} : (!transform.any_op) -> !transform.any_op\n"
            # Step 6: Canonicalize
            f'    %func2 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %func2 : (!transform.any_op) -> !transform.any_op\n'
            # Step 7: Re-tag for downstream
            f'    %func3 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_op = transform.structured.match attributes{{tag = "tiled_target"}} in %func3 : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %final_op "tag" = %final_tag : !transform.any_op, !transform.any_param\n'
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
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
        return [len(cls.OPERAND_COMBOS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        idx = raw_slots[0] % len(cls.OPERAND_COMBOS)
        return {"operands_to_promote": cls.OPERAND_COMBOS[idx]}
