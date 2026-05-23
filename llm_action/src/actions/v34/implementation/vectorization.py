import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Tile to SIMD-friendly sizes for auto-vectorization of pooling operations.

    Pooling_nchw_max has windowed access patterns (oh*stride+kh, ow*stride+kw) that
    prevent direct MLIR-level vectorization even with vectorize_nd_extract. Instead,
    we tile selected dimensions to SIMD-width sizes and generalize the op, allowing
    LLVM's auto-vectorizer to handle the lowered loops efficiently.

    This is a lowering transform: after application, the tiled linalg.generic has
    SIMD-sized inner dimensions that LLVM can vectorize directly.
    """

    unique_execution: bool = True  # one-shot lowering; second application has no valid linalg target inside

    VOCAB = [0, 4, 8, 16, 32]  # 0 = do not tile; SIMD-friendly sizes for f64/f32

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Per-loop tile/vector sizes. 0 = do not tile. At least one must be > 0. Non-zero product must be <= VECTORIZATION_SIZE_LIMIT.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes:
            return False
        if all(s == 0 for s in tile_sizes):
            return False
        if any(s < 0 for s in tile_sizes):
            return False
        product = 1
        for s in tile_sizes:
            if s > 0:
                product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        # Must have a linalg op to vectorize (not already lowered)
        if "linalg." not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_loops = sum(1 for s in tile_sizes if s != 0)
        loop_results = ", ".join(["!transform.any_op"] * n_loops)

        # Tile to SIMD sizes and generalize for LLVM auto-vectorization
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {str(tile_sizes)} : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    %generic = transform.structured.generalize %tiled_op : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        # Clamp product to VECTORIZATION_SIZE_LIMIT
        product = 1
        for s in sizes:
            if s > 0:
                product *= s
        while product > VECTORIZATION_SIZE_LIMIT:
            reduced = False
            for j in range(len(sizes) - 1, -1, -1):
                if sizes[j] == 0:
                    continue
                idx_in_vocab = cls.VOCAB.index(sizes[j]) if sizes[j] in cls.VOCAB else len(cls.VOCAB) - 1
                if idx_in_vocab > 1:  # index 0 is 0 (skip), index 1 is smallest non-zero
                    sizes[j] = cls.VOCAB[idx_in_vocab - 1]
                    reduced = True
                    break
            if not reduced:
                break
            product = 1
            for s in sizes:
                if s > 0:
                    product *= s
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
                slot_mask[0] = True  # fallback to 0 (skip)
            masks.append(slot_mask)
        return np.concatenate(masks)
