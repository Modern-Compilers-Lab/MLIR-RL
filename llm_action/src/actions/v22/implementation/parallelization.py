from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Parallelization(ActionBase):
    """
    Distribute iterations of parallel outer loops across hardware threads
    using tiled partitioning via scf.forall.

    This is a structural transform that introduces scf.forall, replacing the
    original loop structure. A second application has no valid target.
    """

    unique_execution: bool = True  # introduces scf.forall, changes loop kind

    THREAD_OPTIONS = [2, 4, 8, 16, 32]  # multiples of 2 to avoid dynamic shapes

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

        # For matmul-like ops, parallel dims are the first two (M, N).
        # Reduction dim (K) must not be parallelized (use 0).
        # We distribute threads across the first parallel dimension only.
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op:2 = transform.structured.tile_using_forall %op num_threads [{num_threads}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op#1 "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        idx = raw_slots[0] % len(cls.THREAD_OPTIONS)
        return {"num_threads": cls.THREAD_OPTIONS[idx]}
