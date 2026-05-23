import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class ParallelizationByNumThreads(ActionBase):
    """Partition iterations of outer parallel loops across a fixed number of threads.

    Uses tile_using_forall with num_threads. Simpler than tiling-based parallelization
    and sufficient when outer dimensions are large. The num_threads must divide the
    iteration count for safe tile computation (avoids dynamic shape issues in MLIR).

    This is a one-shot transform: introduces scf.forall which changes loop structure.
    """

    unique_execution: bool = True  # introduces scf.forall, second application has no valid target

    THREAD_OPTIONS = [2, 4, 8, 14, 28]  # must be divisors of iteration counts; multiples of 2 preferred

    @classmethod
    def parameters(cls) -> dict:
        return {
            "num_threads": {
                "description": "Number of threads to distribute work across.",
                "type": "int",
                "values": cls.THREAD_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        num_threads = params.get("num_threads")
        if num_threads is None:
            return False
        if num_threads < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        num_threads = params["num_threads"]

        # Apply num_threads to the first (batch N) dimension only
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall_op = transform.structured.tile_using_forall %op num_threads [{num_threads}, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
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
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        return {"num_threads": cls.THREAD_OPTIONS[raw_slots[0] % len(cls.THREAD_OPTIONS)]}
