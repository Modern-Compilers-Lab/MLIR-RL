from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code
import numpy as np


class Parallelization(ActionBase):
    """Distribute the outermost parallel loop across a fixed number of threads
    using scf.forall with num_threads.
    Lowering (Category B): introduces scf.forall, changing the loop structure."""

    # The outermost loop is consumed into forall; a second application has no valid parallel target.
    unique_execution: bool = True

    THREAD_OPTIONS = [2, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "num_threads": {
                "description": "Number of parallel threads for the outermost loop.",
                "type": "int",
                "values": cls.THREAD_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        num_threads = params.get("num_threads", 0)
        if not isinstance(num_threads, int) or num_threads <= 0:
            return False
        if num_threads not in cls.THREAD_OPTIONS:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        num_threads = params["num_threads"]

        transform_code = (
            "module attributes {transform.with_named_sequence} {\n"
            "  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n"
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %forall = transform.structured.tile_using_forall %op num_threads [{num_threads}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
            '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            '    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            "    transform.yield\n"
            "  }\n"
            "}\n"
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
        idx = raw_slots[0] % len(cls.THREAD_OPTIONS)
        return {"num_threads": cls.THREAD_OPTIONS[idx]}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        # num_threads should divide the first loop bound (outermost loop)
        bound = loop_bounds[0] if loop_bounds else 0
        mask = np.array(
            [bound > 0 and bound % t == 0 for t in cls.THREAD_OPTIONS],
            dtype=bool,
        )
        if not mask.any():
            mask[0] = True
        return mask
