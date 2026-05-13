import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


def _count_loops(code: str) -> int:
    if 'tag = "operation_0"' not in code:
        return 0
    if "linalg.matmul" in code:
        return 3
    if "linalg.conv_2d_nchw_fchw" in code:
        return 7
    if "linalg.pooling_nchw_max" in code:
        return 6
    m = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
    if m:
        return len([s.strip() for s in m.group(1).split(',')])
    m = re.search(r'outs\([^:]+:\s*tensor<([^>]+)>', code)
    if m:
        return len(m.group(1).split('x')) - 1
    return 0


class LoopPeeling(ActionBase):
    """Tile a dimension and peel remainder iterations into a cleanup loop.
    Ensures the main loop has a clean trip count for vectorization."""

    unique_execution = False  # Can peel different dimensions at different levels

    VOCAB = [3, 6, 12, 24, 48]

    @classmethod
    def parameters(cls) -> dict:
        return {"tile_size": "tile size for the first dimension; remainder is peeled"}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_size = params.get("tile_size", 0)
        if tile_size < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_size = params["tile_size"]
        n_loops = _count_loops(code)
        if n_loops == 0:
            return code

        # Tile the first dimension with the given size, then peel the resulting loop
        tile_sizes = [tile_size] + [0] * (n_loops - 1)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    %peeled, %remainder = transform.loop.peel %loop {{fail_if_already_divisible = false}} : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            return run_transform_code(code, transform_code)
        except Exception:
            # If peeling fails (already divisible), fall back to just tiling
            n_tiled = sum(1 for s in tile_sizes if s != 0)
            r = ', '.join(['!transform.any_op'] * n_tiled)
            fallback_code = (
                f'module attributes {{transform.with_named_sequence}} {{\n'
                f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
                f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f'    %tiled_op, %loops:{n_tiled} = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {r})\n'
                f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
                f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
                f'    transform.yield\n'
                f'  }}\n'
                f'}}\n'
            )
            try:
                return run_transform_code(code, fallback_code)
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
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {"tile_size": cls.VOCAB[raw_slots[0] % len(cls.VOCAB)]}
