import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Tiling(ActionBase):
    """Partition the iteration space into multi-dimensional tiles for cache locality.
    Mode 0 = sequential (tile_using_for), Mode 1 = parallel (tile_using_forall).
    """

    unique_execution = True  # multi-level tiling is a meaningful tuning strategy

    VOCAB = [0, 4, 8, 16, 32]  # per-loop tile sizes; 0 = do not tile this dimension
    MODE_VOCAB = [0, 1]  # 0 = sequential (tile_using_for), 1 = parallel (tile_using_forall)

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "type": "list[int]",
                "description": "Per-loop tile sizes (0 means do not tile that dimension)",
                "values": cls.VOCAB,
            },
            "parallelize": {
                "type": "int",
                "description": "0 = sequential tiling, 1 = parallel tiling",
                "values": cls.MODE_VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or all(s == 0 for s in tile_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        parallelize = params.get("parallelize", 0)

        n_loops = sum(1 for s in tile_sizes if s != 0)
        if n_loops == 0:
            return code

        if parallelize == 0:
            loop_types = ", ".join(["!transform.any_op"] * n_loops)
            transform_code = (
                f"module attributes {{transform.with_named_sequence}} {{\n"
                f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
                f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {tile_sizes}"
                f" : (!transform.any_op) -> (!transform.any_op, {loop_types})\n"
                f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
                f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
                f"    transform.yield\n"
                f"  }}\n"
                f"}}\n"
            )
        else:
            transform_code = (
                f"module attributes {{transform.with_named_sequence}} {{\n"
                f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
                f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f"    %tiled_op, %forall = transform.structured.tile_using_forall %op tile_sizes {tile_sizes}"
                f" : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
                f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
                f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return [len(cls.MODE_VOCAB)] + [len(cls.VOCAB)] * n

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS - 1)
        parallelize = cls.MODE_VOCAB[raw_slots[0] % len(cls.MODE_VOCAB)]
        sizes = [cls.VOCAB[raw_slots[1 + i] % len(cls.VOCAB)] for i in range(n)]
        return {"tile_sizes": sizes, "parallelize": parallelize}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n = min(n_loops, MAX_PARAM_SLOTS - 1)
        masks = [np.ones(len(cls.MODE_VOCAB), dtype=bool)]  # mode slot: always valid
        for i in range(n):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            m = np.array(
                [s == 0 or (bound > 0 and bound % s == 0) for s in cls.VOCAB],
                dtype=bool,
            )
            if not m.any():
                m[0] = True
            masks.append(m)
        return np.concatenate(masks)
