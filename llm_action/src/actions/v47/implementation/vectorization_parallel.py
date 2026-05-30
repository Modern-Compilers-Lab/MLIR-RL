import re

import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


def _get_iterator_types(code: str) -> list[str] | None:
    """Extract iterator types from linalg op in the MLIR code.

    Returns list of 'parallel' or 'reduction' strings, or None if undetermined.
    """
    # Named linalg ops with known iterator types
    if "linalg.matmul" in code:
        return ["parallel", "parallel", "reduction"]

    # Generic linalg ops: parse iterator_types attribute
    m = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
    if m:
        types_str = m.group(1)
        types = []
        for t in types_str.split(","):
            t = t.strip().strip('"')
            if t in ("parallel", "reduction"):
                types.append(t)
        if types:
            return types

    return None


class VectorizationParallel(ActionBase):
    """Tile inner loops to SIMD-width granularity using parallel forall-loops
    for parallel dims and sequential for-loops for reduction dims.

    LLVM auto-vectorizes the resulting small tile bodies. This combines
    SIMD lowering with thread-level distribution in a single transformation.

    Preprocessing: tile_using_for for reduction dims, then tile_using_forall
    for parallel dims (this order is required because tile_using_for cannot
    work inside scf.forall bodies).

    unique_execution = True: introduces forall and changes the loop structure
    fundamentally; a second application has no valid linalg target.
    """

    unique_execution: bool = True  # lowering: changes loop structure fundamentally

    VOCAB = [1, 4, 8, 16, 32, 64]  # 1 = skip that dimension

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile size per loop dimension; parallel dims get forall, reduction dims get for; 1 means skip.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes")
        if not tile_sizes or not isinstance(tile_sizes, list):
            return False
        if all(s <= 1 for s in tile_sizes):
            return False  # all-skip = no-op
        if any(not isinstance(s, int) or s < 1 for s in tile_sizes):
            return False
        iter_types = _get_iterator_types(code)
        if not iter_types:
            return False
        if len(tile_sizes) != len(iter_types):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        iter_types = _get_iterator_types(code)
        if not iter_types:
            return code

        # Split into reduction and parallel sizes (1 = skip)
        reduction_sizes = [tile_sizes[i] if iter_types[i] == "reduction" and tile_sizes[i] > 1 else 0
                          for i in range(len(tile_sizes))]
        parallel_sizes = [tile_sizes[i] if iter_types[i] == "parallel" and tile_sizes[i] > 1 else 0
                         for i in range(len(tile_sizes))]

        has_reduction = any(s > 0 for s in reduction_sizes)
        has_parallel = any(s > 0 for s in parallel_sizes)

        if not has_reduction and not has_parallel:
            return code

        lines = []
        lines.append('module attributes {transform.with_named_sequence} {')
        lines.append('  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {')
        lines.append('    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op')

        current_op = "%op"
        tag_target = None

        if has_reduction:
            n_red_loops = sum(1 for s in reduction_sizes if s != 0)
            red_loop_results = ", ".join(["!transform.any_op"] * n_red_loops)
            lines.append(
                f'    %red_tiled, %red_loops:{n_red_loops} = transform.structured.tile_using_for {current_op}'
                f' tile_sizes {str(reduction_sizes)} : (!transform.any_op) -> (!transform.any_op, {red_loop_results})'
            )
            current_op = "%red_tiled"
            tag_target = "%red_loops#0"  # outermost reduction loop

        if has_parallel:
            lines.append(
                f'    %par_tiled, %forall = transform.structured.tile_using_forall {current_op}'
                f' tile_sizes {str(parallel_sizes)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)'
            )
            if tag_target is None:
                tag_target = "%forall"  # forall is outermost if no reduction tiling

        # Tag the outermost handle
        lines.append(f'    %tag = transform.param.constant "operation_0" -> !transform.any_param')
        lines.append(f'    transform.annotate {tag_target} "tag" = %tag : !transform.any_op, !transform.any_param')
        lines.append('    transform.yield')
        lines.append('  }')
        lines.append('}')

        transform_code = "\n".join(lines)

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

    # ---- RL interface ----

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
                [s == 1 or (bound > 0 and bound % s == 0) for s in cls.VOCAB],
                dtype=bool,
            )
            if not slot_mask.any():
                slot_mask[0] = True  # 1 is always safe (skip)
            masks.append(slot_mask)
        return np.concatenate(masks)
