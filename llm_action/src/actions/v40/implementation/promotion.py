import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """Copy tiled operand slices into contiguous aligned temporary buffers.
    Internally tiles, bufferizes, promotes, and canonicalizes in one transform sequence.
    """

    unique_execution = True  # involves bufferization which is a one-shot structural change

    OPERAND_OPTIONS = [[0, 1, 2], [0, 1], [0], [1], [2]]
    TILE_VOCAB = [8, 16, 32, 64, 128]  # tile sizes for promotion tiling

    @classmethod
    def parameters(cls) -> dict:
        return {
            "operands_to_promote": {
                "type": "list[int]",
                "description": "Operand indices to copy into contiguous buffers",
                "values": cls.OPERAND_OPTIONS,
            },
            "tile_sizes": {
                "type": "list[int]",
                "description": "Tile sizes for promotion tiling",
                "values": cls.TILE_VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        operands = params.get("operands_to_promote", [])
        tile_sizes = params.get("tile_sizes", [])
        if not operands or not tile_sizes:
            return False
        if all(s == 0 for s in tile_sizes):
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

        loop_types = ", ".join(["!transform.any_op"] * n_loops)
        operands_str = ", ".join(str(o) for o in operands)

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n"
            # Step 1: Match and tile
            f'    %op0 = transform.structured.match attributes{{tag = "operation_0"}} in %module'
            f" : (!transform.any_op) -> !transform.any_op\n"
            f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op0 tile_sizes {tile_sizes}"
            f" : (!transform.any_op) -> (!transform.any_op, {loop_types})\n"
            # Tag the tiled op with intermediate tag for re-matching after bufferization
            f'    %tag_val = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag_val : !transform.any_op, !transform.any_param\n'
            # Step 2: Bufferize (apply directly to consumed module handle)
            f"    %buf = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}}"
            f" %module {{bufferize_function_boundaries = true}} : (!transform.any_op) -> !transform.any_op\n"
            # Step 3: Re-match after bufferization (all prior handles invalidated)
            f'    %func1 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}}'
            f" in %buf : (!transform.any_op) -> !transform.any_op\n"
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}}'
            f" in %func1 : (!transform.any_op) -> !transform.any_op\n"
            # Step 4: Promote with use_alloca (stack allocation avoids buffer-deallocation-pipeline issues)
            f"    %promoted_op = transform.structured.promote %promoted_target"
            f" {{operands_to_promote = [{operands_str}], use_alloca}}"
            f" : (!transform.any_op) -> !transform.any_op\n"
            # Step 5: Canonicalize (fold dynamic shapes to static after promotion)
            f'    %func2 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}}'
            f" in %buf : (!transform.any_op) -> !transform.any_op\n"
            f'    transform.apply_registered_pass "canonicalize" to %func2'
            f" : (!transform.any_op) -> !transform.any_op\n"
            # Step 6: Re-tag for downstream actions
            f'    %func3 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}}'
            f" in %buf : (!transform.any_op) -> !transform.any_op\n"
            f'    %final_op = transform.structured.match attributes{{tag = "tiled_target"}}'
            f" in %func3 : (!transform.any_op) -> !transform.any_op\n"
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
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        n = min(n_loops, MAX_PARAM_SLOTS - 1)
        return [len(cls.OPERAND_OPTIONS)] + [len(cls.TILE_VOCAB)] * n

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS - 1)
        operand_idx = raw_slots[0] % len(cls.OPERAND_OPTIONS)
        operands = cls.OPERAND_OPTIONS[operand_idx]
        sizes = [cls.TILE_VOCAB[raw_slots[1 + i] % len(cls.TILE_VOCAB)] for i in range(n)]
        return {"operands_to_promote": operands, "tile_sizes": sizes}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n = min(n_loops, MAX_PARAM_SLOTS - 1)
        masks = [np.ones(len(cls.OPERAND_OPTIONS), dtype=bool)]  # operand slot: always valid
        for i in range(n):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            m = np.array(
                [bound > 0 and bound % s == 0 for s in cls.TILE_VOCAB],
                dtype=bool,
            )
            if not m.any():
                m[0] = True
            masks.append(m)
        return np.concatenate(masks)
