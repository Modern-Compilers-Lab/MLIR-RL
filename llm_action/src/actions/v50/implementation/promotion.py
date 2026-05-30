import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """Copy tiled operand sub-regions into contiguous temporary buffers (alloca).

    Requires tiling as a prerequisite (done internally). After tiling, bufferizes
    the module, re-matches the tiled op, promotes specified operands, and
    canonicalizes. Uses stack allocation (use_alloca) to avoid deallocation issues.
    Repeatable: can promote different operands or at different tile levels.
    """

    unique_execution: bool = True  # can promote at different tile levels

    TILE_VOCAB = [0, 2, 4, 8, 16, 32]  # tile sizes; 0 = do not tile

    # Operand combinations to promote
    OPERAND_OPTIONS = [
        [0, 1, 2],  # all operands (input, kernel, output)
        [0, 1],     # both inputs
        [0, 2],     # input + output
        [0],        # input only
        [1],        # kernel only
        [2],        # output only
    ]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Per-loop tile sizes for the prerequisite tiling; 0 = skip",
                "type": "list[int]",
                "values": cls.TILE_VOCAB,
            },
            "operands_to_promote": {
                "description": "Indices of operands to promote into contiguous local buffers",
                "type": "list[int]",
                "values": cls.OPERAND_OPTIONS,
            },
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
        operands = params.get("operands_to_promote", [])
        if not operands or not isinstance(operands, list):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        operands = params["operands_to_promote"]

        n_loops = sum(1 for s in tile_sizes if s != 0)
        if n_loops == 0:
            return code

        loop_results = ", ".join(["!transform.any_op"] * n_loops)
        operands_str = str(operands)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n'
            # Step 1: Match and tile
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %module'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op'
            f' tile_sizes {str(tile_sizes)}'
            f' : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            # Tag tiled op so we can find it after bufferization
            f'    %tiled_tag = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tiled_tag : !transform.any_op, !transform.any_param\n'
            # Step 2: Bufferize (invalidates all handles)
            f'    %buf = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}} %module'
            f' {{bufferize_function_boundaries = true}} : (!transform.any_op) -> !transform.any_op\n'
            # Step 3: Re-match after bufferization
            f'    %func1 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}} in %func1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            # Step 4: Promote with stack allocation
            f'    %promoted_op = transform.structured.promote %promoted_target'
            f' {{operands_to_promote = {operands_str}, use_alloca}}'
            f' : (!transform.any_op) -> !transform.any_op\n'
            # Step 5: Canonicalize
            f'    %func2 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %func2 : (!transform.any_op) -> !transform.any_op\n'
            # Step 6: Re-tag for downstream actions
            f'    %func3 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_op = transform.structured.match attributes{{tag = "tiled_target"}} in %func3'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %final_op "tag" = %final_tag : !transform.any_op, !transform.any_param\n'
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
        # tile_sizes (up to MAX_PARAM_SLOTS - 1 slots) + operand_choice (1 slot)
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        # First (n-1) slots: tile sizes per dimension
        # Last slot: operand combination choice
        n_tile_slots = min(n_loops, MAX_PARAM_SLOTS - 1)
        tile_classes = [len(cls.TILE_VOCAB)] * n_tile_slots
        operand_classes = [len(cls.OPERAND_OPTIONS)]
        return tile_classes + operand_classes

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n_tile_slots = min(n_loops, MAX_PARAM_SLOTS - 1)
        tile_sizes = [cls.TILE_VOCAB[raw_slots[i] % len(cls.TILE_VOCAB)] for i in range(n_tile_slots)]
        operand_idx = raw_slots[n_tile_slots] % len(cls.OPERAND_OPTIONS)
        return {
            "tile_sizes": tile_sizes,
            "operands_to_promote": cls.OPERAND_OPTIONS[operand_idx],
        }

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n_tile_slots = min(n_loops, MAX_PARAM_SLOTS - 1)
        masks = []
        for i in range(n_tile_slots):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            slot_mask = np.array(
                [s == 0 or (bound > 0 and bound % s == 0) for s in cls.TILE_VOCAB],
                dtype=bool,
            )
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        # Operand choice slot: all valid
        masks.append(np.ones(len(cls.OPERAND_OPTIONS), dtype=bool))
        return np.concatenate(masks)
