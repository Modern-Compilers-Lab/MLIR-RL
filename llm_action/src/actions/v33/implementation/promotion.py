from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code
import numpy as np


class Promotion(ActionBase):
    """Promote tiled operands into contiguous local buffers via
    tile -> bufferize -> promote -> canonicalize.
    Structure-preserving (Category A): the linalg op remains but operates
    on contiguous promoted buffers."""

    # Promoting different operand subsets is a meaningful tuning knob.
    unique_execution: bool = True

    # Fixed operand promotion options
    OPERAND_OPTIONS = [
        [0, 1, 2],  # all operands
        [0, 1],     # both inputs
        [0, 2],     # input + output
        [1, 2],     # filter + output
        [0],        # input only
    ]

    # Internal tile sizes for creating the subview structure needed by promote.
    # Uses small values (4) that divide most conv2d dimension sizes.
    _INTERNAL_TILE_SIZES_TEMPLATE = [0, 4, 0, 0, 4, 0, 0]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "operands_to_promote": {
                "description": "List of operand indices to copy into contiguous local buffers.",
                "type": "list[int]",
                "values": cls.OPERAND_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        ops = params.get("operands_to_promote", [])
        if not ops or not isinstance(ops, list):
            return False
        if any(not isinstance(o, int) or o < 0 for o in ops):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        operands = params["operands_to_promote"]
        operands_str = ", ".join(str(o) for o in operands)
        tile_sizes = cls._INTERNAL_TILE_SIZES_TEMPLATE

        # Count non-zero tile sizes for loop handles
        n_loops = sum(1 for s in tile_sizes if s != 0)
        result_types = ", ".join(["!transform.any_op"] * (1 + n_loops))

        transform_code = (
            "module attributes {transform.with_named_sequence} {\n"
            "  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.consumed}) {\n"
            # Step 1: Match and tile
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %module : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> ({result_types})\n"
            # Tag tiled op for re-matching after bufferization
            '    %tiled_tag = transform.param.constant "tiled_target" -> !transform.any_param\n'
            '    transform.annotate %tiled_op "tag" = %tiled_tag : !transform.any_op, !transform.any_param\n'
            # Step 2: Bufferize (invalidates ALL handles)
            "    %new_module = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %module {bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op\n"
            # Step 3: Re-match after bufferization
            '    %promoted_target = transform.structured.match attributes{tag = "tiled_target"} in %new_module : (!transform.any_op) -> !transform.any_op\n'
            # Step 4: Promote
            f'    %promoted_op = transform.structured.promote %promoted_target {{operands_to_promote = [{operands_str}], use_alloca}} : (!transform.any_op) -> !transform.any_op\n'
            # Step 5: Canonicalize (fold dynamic shapes to static)
            '    %func2 = transform.structured.match ops{["func.func"]} in %new_module : (!transform.any_op) -> !transform.any_op\n'
            '    transform.apply_registered_pass "canonicalize" to %func2 : (!transform.any_op) -> !transform.any_op\n'
            # Step 6: Re-tag for downstream actions
            '    %final_op = transform.structured.match attributes{tag = "tiled_target"} in %new_module : (!transform.any_op) -> !transform.any_op\n'
            '    %final_tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            '    transform.annotate %final_op "tag" = %final_tag : !transform.any_op, !transform.any_param\n'
            "    transform.yield\n"
            "  }\n"
            "}\n"
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
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        idx = raw_slots[0] % len(cls.OPERAND_OPTIONS)
        return {"operands_to_promote": cls.OPERAND_OPTIONS[idx]}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        return None
