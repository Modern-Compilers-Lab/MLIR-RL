import re

import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


def _get_n_loops(code: str) -> int:
    """Extract number of loop dimensions from iterator_types in the code."""
    match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
    if not match:
        return 0
    return len([t.strip() for t in match.group(1).split(',')])


def _get_first_loop_bound(code: str) -> int:
    """Extract the first dimension size from the tensor type in the code."""
    match = re.search(r'tensor<(\d+)', code)
    if match:
        return int(match.group(1))
    return 0


class ParallelizationThreads(ActionBase):
    """Distribute iterations across threads using tile_using_forall with num_threads.

    Structure-preserving (Category A): result is a tiled linalg.* inside scf.forall.
    Partitions the first parallel dimension among the specified number of threads.
    Tags the tiled linalg op.
    """

    THREAD_OPTIONS = [2, 4, 8, 16, 32, 64]
    unique_execution = True  # Introducing scf.forall changes the loop kind; one-shot parallelization

    @classmethod
    def parameters(cls) -> dict:
        return {
            "num_threads": {
                "description": "Number of threads to distribute across.",
                "type": "int",
                "values": cls.THREAD_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        num_threads = params.get("num_threads", 0)
        if num_threads <= 1:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        num_threads = params["num_threads"]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op:2 = transform.structured.tile_using_forall %op num_threads [{num_threads}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        if 'func.func' not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.THREAD_OPTIONS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        return {"num_threads": cls.THREAD_OPTIONS[raw_slots[0] % len(cls.THREAD_OPTIONS)]}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds or len(loop_bounds) == 0:
            return None
        first_bound = loop_bounds[0]
        mask = np.array([
            first_bound > 0 and first_bound % t == 0
            for t in cls.THREAD_OPTIONS
        ], dtype=bool)
        if not mask.any():
            mask[0] = True
        return mask
