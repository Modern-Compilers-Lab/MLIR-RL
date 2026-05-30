import re
import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class VectorizationParallel(ActionBase):
    """Tile innermost dims using tile_using_for for reduction dims and
    tile_using_forall for parallel dims, distributing outer tiles across threads.
    LLVM auto-vectorizes the small inner tiles.

    unique_execution = True: introduces scf.forall and changes loop structure;
    a second application on the same target is ill-defined."""

    unique_execution: bool = True  # changes loop structure fundamentally

    VOCAB = [1, 2, 4, 8, 16, 32]  # tile sizes per dimension

    # Iterator type patterns for known op families
    _ITERATOR_TYPES = {
        "matmul": ["parallel", "parallel", "reduction"],
        "conv_2d_nchw_fchw": ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"],
        "pooling_nchw_max": ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"],
        "add": ["parallel", "parallel", "parallel", "parallel"],
    }

    @classmethod
    def _detect_iterator_types(cls, code: str, n_loops: int) -> list[str]:
        """Detect iterator types from the IR."""
        # Check for known named ops
        if "linalg.matmul" in code:
            return cls._ITERATOR_TYPES["matmul"]
        if "linalg.conv_2d_nchw_fchw" in code:
            return cls._ITERATOR_TYPES["conv_2d_nchw_fchw"]
        if "linalg.pooling_nchw_max" in code:
            return cls._ITERATOR_TYPES["pooling_nchw_max"]
        if "linalg.add" in code:
            return cls._ITERATOR_TYPES["add"]

        # Parse iterator_types from generic
        match = re.search(r'iterator_types\s*=\s*\[(.*?)\]', code)
        if match:
            types_str = match.group(1)
            types = [t.strip().strip('"') for t in types_str.split(',')]
            return types

        # Default: all parallel
        return ["parallel"] * n_loops

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile size per loop dimension for parallel distribution",
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
        if all(s <= 1 for s in tile_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_dims = len(tile_sizes)
        iterator_types = cls._detect_iterator_types(code, n_dims)

        # Separate reduction and parallel tile sizes
        reduction_sizes = []
        parallel_sizes = []
        for i, s in enumerate(tile_sizes):
            if i < len(iterator_types) and iterator_types[i] == "reduction":
                reduction_sizes.append(s)
                parallel_sizes.append(0)
            else:
                reduction_sizes.append(0)
                parallel_sizes.append(s)

        # Step 1: tile reduction dims with tile_using_for (if any non-zero)
        has_reduction_tile = any(s > 0 and s != 0 for s in reduction_sizes)
        # Step 2: tile parallel dims with tile_using_forall (if any non-zero/non-1)
        has_parallel_tile = any(s > 1 for s in parallel_sizes)

        if not has_reduction_tile and not has_parallel_tile:
            return code

        lines = []
        lines.append('module attributes {transform.with_named_sequence} {')
        lines.append('  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {')
        lines.append('    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op')

        current_op = "%op"

        if has_reduction_tile:
            n_red_loops = sum(1 for s in reduction_sizes if s != 0)
            red_loop_results = ", ".join(["!transform.any_op"] * n_red_loops)
            lines.append(f'    %red_tiled, %red_loops:{n_red_loops} = transform.structured.tile_using_for {current_op}'
                        f' tile_sizes {str(reduction_sizes)} : (!transform.any_op) -> (!transform.any_op, {red_loop_results})')
            current_op = "%red_tiled"

        if has_parallel_tile:
            # For forall, use num_threads approach: set 0 for dims we don't parallelize
            forall_sizes = [s if s > 1 else 0 for s in parallel_sizes]
            n_par_loops = sum(1 for s in forall_sizes if s != 0)
            par_loop_results = ", ".join(["!transform.any_op"] * n_par_loops)
            lines.append(f'    %par_tiled, %forall = transform.structured.tile_using_forall {current_op}'
                        f' tile_sizes {str(forall_sizes)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)')
            current_op = "%par_tiled"

        # Tag the result - if we have forall, tag the forall; if only for, tag outermost loop
        lines.append(f'    %tag = transform.param.constant "operation_0" -> !transform.any_param')
        if has_parallel_tile:
            lines.append(f'    transform.annotate %forall "tag" = %tag : !transform.any_op, !transform.any_param')
        elif has_reduction_tile:
            lines.append(f'    transform.annotate %red_loops#0 "tag" = %tag : !transform.any_op, !transform.any_param')

        lines.append('    transform.yield')
        lines.append('  }')
        lines.append('}')

        transform_code = '\n'.join(lines) + '\n'

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
                slot_mask[0] = True
            masks.append(slot_mask)
        return np.concatenate(masks)
