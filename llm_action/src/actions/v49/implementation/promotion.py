import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """Copy tiled operand slices into contiguous temporary buffers.

    Eliminates strided memory access within tiles. Requires tiling as a
    preprocessing step (tiles internally with provided tile sizes), then
    bufferizes and promotes. Repeatable: can promote different operands
    at different tiling levels.
    """

    unique_execution: bool = True  # involves bufferization which changes IR form irreversibly

    TILE_VOCAB = [0, 4, 8, 16, 32, 64]
    OPERAND_CONFIGS = [
        [0, 1, 2],  # all operands
        [0, 1],     # inputs only
        [0],        # input A only
        [1],        # input B only
        [2],        # output only
    ]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for the outer tiling before promotion. 0 means no tile on that dim.",
                "type": "list[int]",
                "values": cls.TILE_VOCAB,
            },
            "operands_to_promote": {
                "description": "Which operand indices to promote into contiguous buffers.",
                "type": "list[int]",
                "values": cls.OPERAND_CONFIGS,
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
        if any(s < 0 for s in tile_sizes):
            return False
        # Promotion works on memref (bufferized) form — we bufferize internally.
        # But if the code is already bufferized, skip (to avoid double bufferization issues).
        if "memref<" in code and "tensor<" not in code:
            return False
        operands = params.get("operands_to_promote", [0, 1, 2])
        if not isinstance(operands, list) or not operands:
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
        if n_loops == 0:
            return code

        loop_results = ", ".join(["!transform.any_op"] * n_loops)
        operands_str = ", ".join(str(o) for o in operands)

        # The promotion sequence:
        # 1. Match and tile the op
        # 2. Tag the tiled op so we can find it after bufferization
        # 3. Bufferize (invalidates all handles)
        # 4. Re-match after bufferization
        # 5. Promote
        # 6. Canonicalize
        # 7. Re-tag for downstream actions
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %op0 = transform.structured.match attributes{{tag = "operation_0"}} in %module : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op0 tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    %tag_val = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag_val : !transform.any_op, !transform.any_param\n'
            f'    %buf = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}} %module {{bufferize_function_boundaries = true}} : (!transform.any_op) -> !transform.any_op\n'
            f'    %func1 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}} in %func1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted_op = transform.structured.promote %promoted_target {{operands_to_promote = [{operands_str}], use_alloca}} : (!transform.any_op) -> !transform.any_op\n'
            f'    %func2 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %func2 : (!transform.any_op) -> !transform.any_op\n'
            f'    %func3 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_op = transform.structured.match attributes{{tag = "tiled_target"}} in %func3 : (!transform.any_op) -> !transform.any_op\n'
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
        # tile_sizes (up to MAX_PARAM_SLOTS - 1 slots) + 1 slot for operand config
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        n_tile_slots = min(n_loops, MAX_PARAM_SLOTS - 1)
        tile_classes = [len(cls.TILE_VOCAB)] * n_tile_slots
        operand_classes = [len(cls.OPERAND_CONFIGS)]
        return tile_classes + operand_classes

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n_tile_slots = min(n_loops, MAX_PARAM_SLOTS - 1)
        sizes = [cls.TILE_VOCAB[raw_slots[i] % len(cls.TILE_VOCAB)] for i in range(n_tile_slots)]
        operand_idx = raw_slots[n_tile_slots] % len(cls.OPERAND_CONFIGS)
        return {
            "tile_sizes": sizes,
            "operands_to_promote": cls.OPERAND_CONFIGS[operand_idx],
        }

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n_tile_slots = min(n_loops, MAX_PARAM_SLOTS - 1)
        masks = []
        for i in range(n_tile_slots):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            slot_mask = np.array([
                s == 0 or (bound > 0 and bound % s == 0)
                for s in cls.TILE_VOCAB
            ], dtype=bool)
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        # operand config slot — all valid
        masks.append(np.ones(len(cls.OPERAND_CONFIGS), dtype=bool))
        return np.concatenate(masks)
