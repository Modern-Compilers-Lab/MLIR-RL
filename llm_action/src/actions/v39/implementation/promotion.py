from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code
import numpy as np


class Promotion(ActionBase):
    """Copy tiled operand slices into contiguous local buffers to eliminate strided
    access. Includes internal tiling + bufferization as preprocessing.
    Repeatable: promotion at different tile levels or for different operands is valid."""

    unique_execution: bool = True  # can promote at different tile levels

    TILE_VOCAB = [0, 4, 8, 16, 32]  # 0 = do not tile
    OPERAND_VOCAB = [0, 1, 2, 3]  # 0=[0,1,2], 1=[0,1], 2=[0], 3=[1]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for the outer tiling before promotion; 0 = no tile.",
                "type": "list[int]",
                "values": cls.TILE_VOCAB,
            },
            "operands_to_promote": {
                "description": "Which operands to promote: 0=[0,1,2], 1=[0,1], 2=[0], 3=[1].",
                "type": "int",
                "values": cls.OPERAND_VOCAB,
            },
        }

    @classmethod
    def _decode_operands(cls, operand_choice: int) -> list[int]:
        mapping = {0: [0, 1, 2], 1: [0, 1], 2: [0], 3: [1]}
        return mapping.get(operand_choice, [0, 1, 2])

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
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        operand_choice = params.get("operands_to_promote", 0)
        operands = cls._decode_operands(operand_choice)

        n_loops = sum(1 for s in tile_sizes if s != 0)
        loop_results = ", ".join(["!transform.any_op"] * n_loops)
        operands_str = ", ".join(str(o) for o in operands)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n'
            f'    // Step 1: Match and tile\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %module : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'\n'
            f'    // Tag the tiled op so we can find it after bufferization\n'
            f'    %tag_val = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag_val : !transform.any_op, !transform.any_param\n'
            f'\n'
            f'    // Step 2: Bufferize\n'
            f'    %buf = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}} %module {{bufferize_function_boundaries = true}} : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 3: Re-match after bufferization\n'
            f'    %func1 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}} in %func1 : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 4: Promote\n'
            f'    %promoted_op = transform.structured.promote %promoted_target {{operands_to_promote = [{operands_str}], use_alloca}} : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 5: Canonicalize\n'
            f'    %func2 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %func2 : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    // Step 6: Re-tag for downstream actions\n'
            f'    %func3 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_op = transform.structured.match attributes{{tag = "tiled_target"}} in %func3 : (!transform.any_op) -> !transform.any_op\n'
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
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        n = min(n_loops, MAX_PARAM_SLOTS)
        # First n slots: tile sizes; we don't add the operand slot to keep within MAX_PARAM_SLOTS
        # Instead, we use the last slot for operand selection when possible
        if n < MAX_PARAM_SLOTS:
            return [len(cls.TILE_VOCAB)] * n + [len(cls.OPERAND_VOCAB)]
        else:
            # Use n-1 tile slots + 1 operand slot
            return [len(cls.TILE_VOCAB)] * (n - 1) + [len(cls.OPERAND_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        if n < MAX_PARAM_SLOTS:
            tile_n = n
            sizes = [cls.TILE_VOCAB[raw_slots[i] % len(cls.TILE_VOCAB)] for i in range(tile_n)]
            operand_choice = cls.OPERAND_VOCAB[raw_slots[tile_n] % len(cls.OPERAND_VOCAB)]
        else:
            tile_n = n - 1
            sizes = [cls.TILE_VOCAB[raw_slots[i] % len(cls.TILE_VOCAB)] for i in range(tile_n)]
            operand_choice = cls.OPERAND_VOCAB[raw_slots[tile_n] % len(cls.OPERAND_VOCAB)]
        return {"tile_sizes": sizes, "operands_to_promote": operand_choice}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n = min(n_loops, MAX_PARAM_SLOTS)
        masks = []
        if n < MAX_PARAM_SLOTS:
            tile_n = n
        else:
            tile_n = n - 1
        for i in range(tile_n):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            slot_mask = np.array(
                [s == 0 or (bound > 0 and bound % s == 0) for s in cls.TILE_VOCAB],
                dtype=bool,
            )
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        # Operand slot: all choices always valid
        masks.append(np.ones(len(cls.OPERAND_VOCAB), dtype=bool))
        return np.concatenate(masks)
