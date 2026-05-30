import numpy as np
import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class ParallelizationTiling(ActionBase):
    """Tile outer parallel dimensions and distribute via forall for multi-core execution.

    Uses tile_using_forall with tile_sizes. Only parallel dimensions (non-reduction)
    should be tiled; reduction dims are auto-zeroed in the transform to avoid
    incorrect parallel reduction.

    Lowering transform: introduces scf.forall which changes the loop structure.
    A second application would try to parallelize already-parallel forall loops.
    """

    # Introduces scf.forall — second application has no valid linalg target
    unique_execution: bool = True

    VOCAB = [0, 4, 8, 16, 32, 64]  # 0 = do not parallelize that dimension

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile size per loop dimension for parallel distribution; "
                               "0 means skip. Reduction dims are auto-zeroed.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def _detect_reduction_dims(cls, code: str) -> list[int]:
        """Detect which iterator dimensions are reductions from linalg op attributes."""
        # Look for iterator_types in linalg.generic or infer from named ops
        # For linalg.matmul, dim 2 (K) is the reduction dim
        if "linalg.matmul" in code:
            return [2]
        # For linalg.generic, parse iterator_types
        match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
        if match:
            types = [t.strip().strip('"') for t in match.group(1).split(",")]
            return [i for i, t in enumerate(types) if t == "reduction"]
        return []

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes")
        if not tile_sizes or not isinstance(tile_sizes, list):
            return False
        if any(not isinstance(s, int) or s < 0 for s in tile_sizes):
            return False

        # Zero out reduction dims before checking all-zero
        reduction_dims = cls._detect_reduction_dims(code)
        effective = list(tile_sizes)
        for rd in reduction_dims:
            if rd < len(effective):
                effective[rd] = 0

        if all(s == 0 for s in effective):
            return False  # no parallel dims to tile
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = list(params["tile_sizes"])

        # Auto-zero reduction dims to prevent incorrect parallel reduction
        reduction_dims = cls._detect_reduction_dims(code)
        for rd in reduction_dims:
            if rd < len(tile_sizes):
                tile_sizes[rd] = 0

        if all(s == 0 for s in tile_sizes):
            return code

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall = transform.structured.tile_using_forall %op'
            f' tile_sizes {str(tile_sizes)}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
            slot_mask = np.array([
                s == 0 or (bound > 0 and bound % s == 0)
                for s in cls.VOCAB
            ], dtype=bool)
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        return np.concatenate(masks)
