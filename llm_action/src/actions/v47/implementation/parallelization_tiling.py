import re

import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


def _get_iterator_types(code: str) -> list[str] | None:
    """Extract iterator types from linalg op in the MLIR code."""
    if "linalg.matmul" in code:
        return ["parallel", "parallel", "reduction"]
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


class ParallelizationTiling(ActionBase):
    """Tile the parallel dimensions of the loop nest and distribute via forall.

    Reduction dimensions are automatically set to 0 (not tiled) since
    tile_using_forall cannot tile reduction dimensions. This is a
    structure-preserving transform (Category A): the inner op remains
    a linalg op.

    unique_execution = True: introduces scf.forall which changes the
    loop kind; subsequent transforms cannot operate inside forall bodies.
    """

    unique_execution: bool = True  # introduces forall; second application has no valid target

    VOCAB = [0, 4, 8, 16, 32, 64]  # 0 = do not tile that dimension

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile size per loop dimension for parallel distribution; 0 means skip; reduction dims are auto-zeroed.",
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
        if any(not isinstance(s, int) or s < 0 for s in tile_sizes):
            return False
        iter_types = _get_iterator_types(code)
        if not iter_types:
            return False
        if len(tile_sizes) != len(iter_types):
            return False
        # Zero out reduction dims and check if any parallel dim is tiled
        effective = [tile_sizes[i] if iter_types[i] == "parallel" else 0
                    for i in range(len(tile_sizes))]
        if all(s == 0 for s in effective):
            return False  # no parallel dims tiled = no-op
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

        # Zero out reduction dims
        effective = [tile_sizes[i] if iter_types[i] == "parallel" else 0
                    for i in range(len(tile_sizes))]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall = transform.structured.tile_using_forall %op'
            f' tile_sizes {str(effective)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
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
                [s == 0 or (bound > 0 and bound % s == 0) for s in cls.VOCAB],
                dtype=bool,
            )
            if not slot_mask.any():
                slot_mask[0] = True  # 0 = no-tile always valid
            masks.append(slot_mask)
        return np.concatenate(masks)
