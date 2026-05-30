import re
import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


# Known iterator types per op family (parallel=P, reduction=R)
_ITERATOR_TYPES = {
    "linalg.matmul": ["parallel", "parallel", "reduction"],
    "linalg.conv_2d_nchw_fchw": ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"],
    "linalg.pooling_nchw_max": ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"],
    "linalg.add": ["parallel", "parallel", "parallel", "parallel"],
}


def _parse_iterator_types(code: str) -> list[str] | None:
    """Parse iterator_types from linalg.generic in code."""
    m = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
    if m:
        types = re.findall(r'"(\w+)"', m.group(1))
        return types
    # Try to detect from known op types
    for op_name, types in _ITERATOR_TYPES.items():
        if op_name in code:
            return types
    return None


class VectorizationParallel(ActionBase):
    """Tile loops to SIMD sizes with parallel forall-loop distribution across threads.

    Parallel dimensions use tile_using_forall (thread-distributed), reduction dimensions
    use tile_using_for. LLVM auto-vectorizes the small inner tiles without explicit
    transform.structured.vectorize (which doesn't work inside scf.forall bodies).
    """

    unique_execution = True  # Lowering: introduces scf.forall, changes loop structure

    VOCAB = [1, 2, 4, 8, 16, 32]  # tile sizes per dim

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile size per loop dimension. 1 = scalar. Parallel dims use forall, reduction dims use for.",
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
        if all(s == 1 for s in tile_sizes):
            return False
        if "linalg." not in code:
            return False
        iter_types = _parse_iterator_types(code)
        if iter_types is None:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        iter_types = _parse_iterator_types(code)
        if iter_types is None:
            return code

        n_dims = min(len(tile_sizes), len(iter_types))

        # Split into reduction tile sizes and parallel tile sizes
        reduction_sizes = []
        parallel_sizes = []
        for i in range(n_dims):
            if iter_types[i] == "reduction":
                reduction_sizes.append(tile_sizes[i])
                parallel_sizes.append(0)  # don't parallelize reduction dims
            else:
                reduction_sizes.append(0)  # don't tile reduction dims in for-loop
                parallel_sizes.append(tile_sizes[i])

        # Pad with zeros if needed
        while len(reduction_sizes) < len(iter_types):
            reduction_sizes.append(0)
            parallel_sizes.append(0)

        has_reduction = any(s != 0 for s in reduction_sizes)
        has_parallel = any(s != 0 for s in parallel_sizes)

        if not has_reduction and not has_parallel:
            return code

        # Build transform code
        lines = [
            'module attributes {transform.with_named_sequence} {',
            '  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {',
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op',
        ]

        current_op = "%op"

        # Step 1: Tile reduction dimensions with tile_using_for (must come BEFORE forall)
        if has_reduction:
            n_red_loops = sum(1 for s in reduction_sizes if s != 0)
            red_loop_results = ", ".join(["!transform.any_op"] * n_red_loops)
            lines.append(
                f'    %red_tiled, %red_loops:{n_red_loops} = transform.structured.tile_using_for {current_op} tile_sizes {reduction_sizes} '
                f': (!transform.any_op) -> (!transform.any_op, {red_loop_results})'
            )
            current_op = "%red_tiled"

        # Step 2: Tile parallel dimensions with tile_using_forall
        if has_parallel:
            lines.append(
                f'    %par_tiled:2 = transform.structured.tile_using_forall {current_op} tile_sizes {parallel_sizes} '
                f': (!transform.any_op) -> (!transform.any_op, !transform.any_op)'
            )
            # Tag the forall op (outermost generated structure)
            lines.append(f'    %tag = transform.param.constant "operation_0" -> !transform.any_param')
            lines.append(f'    transform.annotate %par_tiled#1 "tag" = %tag : !transform.any_op, !transform.any_param')
        elif has_reduction:
            # Only reduction, tag the outermost reduction loop
            lines.append(f'    %tag = transform.param.constant "operation_0" -> !transform.any_param')
            lines.append(f'    transform.annotate %red_loops#0 "tag" = %tag : !transform.any_op, !transform.any_param')

        lines.extend([
            '    transform.yield',
            '  }',
            '}',
        ])

        transform_code = "\n".join(lines) + "\n"

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
                bound > 0 and bound % s == 0
                for s in cls.VOCAB
            ], dtype=bool)
            if not slot_mask.any():
                slot_mask[0] = True  # 1 always valid
            masks.append(slot_mask)
        return np.concatenate(masks)
