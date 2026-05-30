import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """Copy tiled operand slices into contiguous local buffers to eliminate
    strided access.

    Internally: tile -> tag -> bufferize -> re-match -> promote -> canonicalize
    -> re-tag.  Uses use_alloca to avoid buffer-deallocation-pipeline issues.

    Repeated application with different tile sizes is meaningful, so
    unique_execution is False.
    """

    unique_execution: bool = True  # different tile sizes / operand subsets

    VOCAB = [0, 32, 64, 128, 256, 512]  # 0 = do not tile that dimension

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for the prerequisite tiling; 0 = skip dimension",
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
        n_loops = sum(1 for s in tile_sizes if s != 0)

        loop_names = ", ".join(f"%loop{i}" for i in range(n_loops))
        loop_types = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n'
            # Step 1: Match and tile
            f'    %op0 = transform.structured.match attributes{{tag = "operation_0"}} in %module'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, {loop_names} = transform.structured.tile_using_for %op0'
            f' tile_sizes {str(tile_sizes)}'
            f' : (!transform.any_op) -> (!transform.any_op, {loop_types})\n'
            # Step 2: Tag tiled op for re-matching after bufferization
            f'    %tag_val = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag_val : !transform.any_op, !transform.any_param\n'
            # Step 3: Bufferize (invalidates ALL prior handles)
            f'    %buf = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}} %module'
            f' {{bufferize_function_boundaries = true}} : (!transform.any_op) -> !transform.any_op\n'
            # Step 4: Re-match after bufferization
            f'    %func1 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}} in %func1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            # Step 5: Promote all operands into contiguous alloca buffers
            f'    %promoted_op = transform.structured.promote %promoted_target'
            f' {{operands_to_promote = [0, 1, 2], use_alloca}}'
            f' : (!transform.any_op) -> !transform.any_op\n'
            # Step 6: Canonicalize to fold dynamic shapes to static
            f'    %func2 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %func2 : (!transform.any_op) -> !transform.any_op\n'
            # Step 7: Re-tag for downstream actions
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
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        n = min(n_loops, MAX_PARAM_SLOTS)
        return [len(cls.VOCAB)] * n

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"tile_sizes": sizes}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n = min(n_loops, MAX_PARAM_SLOTS)
        masks = []
        for i in range(n):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            slot_mask = np.array(
                [s == 0 or (bound > 0 and bound % s == 0) for s in cls.VOCAB],
                dtype=bool,
            )
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        return np.concatenate(masks)
