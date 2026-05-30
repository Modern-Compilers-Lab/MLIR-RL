import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """Tile + bufferize + promote operand slices into contiguous temporary buffers.

    Structure-preserving (Category A): the output is still a linalg op operating on promoted buffers.
    Requires bufferization as prerequisite — handles the full sequence internally.
    Repeated application is meaningful (e.g., promote different operands at different tile levels).
    """

    # Promotion can be applied at different tile levels — repeatable.
    unique_execution: bool = True

    TILE_VOCAB = [0, 16, 32, 64, 128]  # tile sizes for outer blocking before promotion
    OPERAND_OPTIONS = [
        [0, 1, 2],  # all operands
        [0, 1],     # inputs only
        [0],        # first input only
        [1],        # second input only
        [2],        # output only
    ]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for outer blocking before promotion. 0 = do not tile.",
                "type": "list[int]",
                "values": cls.TILE_VOCAB,
            },
            "operands_to_promote": {
                "description": "Which operand indices to promote into contiguous buffers.",
                "type": "list[int]",
                "values": cls.OPERAND_OPTIONS,
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
        if any(not isinstance(s, int) or s < 0 for s in tile_sizes):
            return False
        operands = params.get("operands_to_promote", [0, 1, 2])
        if not operands or not isinstance(operands, list):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        operands = params.get("operands_to_promote", [0, 1, 2])

        n_loops = sum(1 for s in tile_sizes if s != 0)
        loop_results = ", ".join(["!transform.any_op"] * n_loops)
        operands_str = ", ".join(str(o) for o in operands)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n'
            f'\n'
            f'    // Step 1: Match and tile\n'
            f'    %op0 = transform.structured.match attributes{{tag = "operation_0"}} in %module'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op0'
            f' tile_sizes {str(tile_sizes)}'
            f' : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'\n'
            f'    // Tag the tiled op so we can find it after bufferization\n'
            f'    %tag_val = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag_val : !transform.any_op, !transform.any_param\n'
            f'\n'
            f'    // Step 2: Bufferize (invalidates ALL handles)\n'
            f'    %buf = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}}'
            f' %module {{bufferize_function_boundaries = true}}'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 3: Re-match after bufferization\n'
            f'    %func1 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}} in %func1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 4: Promote\n'
            f'    %promoted_op = transform.structured.promote %promoted_target'
            f' {{operands_to_promote = [{operands_str}], use_alloca}}'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 5: Canonicalize\n'
            f'    %func2 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %func2 : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 6: Re-tag for downstream actions\n'
            f'    %func3 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_op = transform.structured.match attributes{{tag = "tiled_target"}} in %func3'
            f' : (!transform.any_op) -> !transform.any_op\n'
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
        # tile_sizes (up to MAX_PARAM_SLOTS - 1 slots) + 1 slot for operand selection
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        n_tile = min(n_loops, MAX_PARAM_SLOTS - 1)
        slots = [len(cls.TILE_VOCAB)] * n_tile
        slots.append(len(cls.OPERAND_OPTIONS))  # operand selection slot
        return slots

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n_tile = min(n_loops, MAX_PARAM_SLOTS - 1)
        tile_sizes = [cls.TILE_VOCAB[raw_slots[i] % len(cls.TILE_VOCAB)] for i in range(n_tile)]
        operand_idx = raw_slots[n_tile] % len(cls.OPERAND_OPTIONS)
        return {
            "tile_sizes": tile_sizes,
            "operands_to_promote": cls.OPERAND_OPTIONS[operand_idx],
        }

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n_tile = min(n_loops, MAX_PARAM_SLOTS - 1)
        masks = []
        for i in range(n_tile):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            slot_mask = np.array([
                s == 0 or (bound > 0 and bound % s == 0)
                for s in cls.TILE_VOCAB
            ], dtype=bool)
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        # Operand selection slot — all always valid
        masks.append(np.ones(len(cls.OPERAND_OPTIONS), dtype=bool))
        return np.concatenate(masks)
