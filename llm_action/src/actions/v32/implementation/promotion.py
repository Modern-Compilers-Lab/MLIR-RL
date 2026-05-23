import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """
    Copy tiled operand data into contiguous local temporary buffers (alloca) before
    the inner computation, eliminating non-unit strides from tiled subviews.
    Repeated at different tile levels is meaningful, so unique_execution = False.
    """

    unique_execution: bool = True

    VOCAB = [0, 16, 32, 64, 128]  # 0 = do not tile this dimension

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Per-loop tile sizes for the initial tiling step before promotion. 0 = no tile.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        # Promotion requires bufferization; skip if already bufferized (memref args)
        # to avoid double-bufferization errors. Allow non-bufferized (tensor) code only.
        import re
        main_match = re.search(r'func\.func @main\(([^)]*)\)', code)
        if main_match and 'memref<' in main_match.group(1):
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes:
            return False
        if all(s == 0 for s in tile_sizes):
            return False
        if any(s < 0 for s in tile_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_nonzero = sum(1 for s in tile_sizes if s != 0)
        if n_nonzero == 0:
            return code

        loop_types = ', '.join(['!transform.any_op'] * n_nonzero)
        sizes_str = str(tile_sizes)

        if n_nonzero == 1:
            loops_lhs = "%tiled_op, %loop"
            loops_types = f"(!transform.any_op, !transform.any_op)"
        else:
            loops_lhs = f"%tiled_op, %loops:{n_nonzero}"
            loops_types = f"(!transform.any_op, {loop_types})"

        transform_code = (
            f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %module'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    {loops_lhs} = transform.structured.tile_using_for %op tile_sizes {sizes_str}'
            f' : (!transform.any_op) -> {loops_types}\n'
            f'    %tmp_tag = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tmp_tag : !transform.any_op, !transform.any_param\n'
            f'\n'
            f'    %new_module = transform.bufferization.one_shot_bufferize'
            f' layout{{IdentityLayoutMap}} %module {{bufferize_function_boundaries = true}}'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}}'
            f' in %new_module : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted_op = transform.structured.promote %promoted_target'
            f' {{operands_to_promote = [0, 1, 2], use_alloca}} : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    %func2 = transform.structured.match ops{{["func.func"]}} in %new_module'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %func2 : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    %final_op = transform.structured.match attributes{{tag = "tiled_target"}}'
            f' in %new_module : (!transform.any_op) -> !transform.any_op\n'
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
        # Promotion produces bufferized code: expect memref.alloca
        if "memref.alloca" not in after and "memref.alloc" not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

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
            slot_mask = np.array([
                s == 0 or (bound > 0 and bound % s == 0)
                for s in cls.VOCAB
            ], dtype=bool)
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        return np.concatenate(masks)
