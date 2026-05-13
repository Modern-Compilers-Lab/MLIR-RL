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


class LoopUnrolling(ActionBase):
    """Unroll the outermost loop of a tiled operation to expose ILP.
    Internally tiles the first dimension, then unrolls the resulting loop."""

    unique_execution = False  # Can unroll different loops at different levels

    VOCAB = [2, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {"unroll_factor": "integer factor to unroll the loop by"}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        factor = params.get("unroll_factor", 0)
        if factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        factor = params["unroll_factor"]
        n_loops = _count_loops(code)
        if n_loops == 0:
            return code

        # Tile the first dimension with size 32 to create a loop, then unroll it
        tile_sizes = [32] + [0] * (n_loops - 1)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.loop.unroll %loop {{factor = {factor} : i64}} : !transform.any_op\n'
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
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {"unroll_factor": cls.VOCAB[raw_slots[0] % len(cls.VOCAB)]}
