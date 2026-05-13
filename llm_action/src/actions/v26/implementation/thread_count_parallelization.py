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


class ThreadCountParallelization(ActionBase):
    """Distribute loop iterations across a fixed number of threads via scf.forall.
    The number of threads must divide the first parallel dimension."""

    unique_execution = True  # Introduces scf.forall; second application has no valid linalg target

    THREAD_OPTIONS = [2, 4, 8, 14, 28]

    @classmethod
    def parameters(cls) -> dict:
        return {"num_threads": "number of worker threads for the first parallel dimension"}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        num_threads = params.get("num_threads", 0)
        if num_threads < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        num_threads = params["num_threads"]
        n_loops = _count_loops(code)
        if n_loops == 0:
            return code

        # Apply num_threads only to the first dimension, 0 for the rest
        thread_list = [num_threads] + [0] * (n_loops - 1)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall = transform.structured.tile_using_forall %op num_threads {thread_list} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
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
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.THREAD_OPTIONS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {"num_threads": cls.THREAD_OPTIONS[raw_slots[0] % len(cls.THREAD_OPTIONS)]}
