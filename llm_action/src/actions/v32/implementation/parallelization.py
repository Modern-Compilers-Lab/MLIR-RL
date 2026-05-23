import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Parallelization(ActionBase):
    """
    Annotate outer parallel loops for multi-threaded execution by distributing
    loop iterations across available cores using scf.forall.
    This introduces scf.forall and changes the loop kind — a second application
    has no valid linalg target — so unique_execution = True.
    """

    # unique_execution = True: creates scf.forall (one-shot parallel lowering);
    # a second application cannot find the original linalg target.
    unique_execution: bool = True

    # Thread configurations (M_threads, N_threads) for 28-core Broadwell target.
    # Product = 28 cores; covers different M/N split strategies.
    THREAD_CONFIGS = [(28, 1), (14, 2), (7, 4), (4, 7), (2, 14)]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "num_threads": {
                "description": (
                    "Number of threads per parallel dimension [M, N]. "
                    "Drawn from THREAD_CONFIGS: (28,1), (14,2), (7,4), (4,7), (2,14)."
                ),
                "type": "list[int]",
                "values": [list(cfg) for cfg in cls.THREAD_CONFIGS],
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        num_threads = params.get("num_threads", [])
        if not num_threads:
            return False
        if any(t <= 0 for t in num_threads):
            return False
        # At least one dimension must use > 1 thread
        if all(t == 1 for t in num_threads):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        num_threads = params["num_threads"]
        threads_str = str(num_threads)

        transform_code = (
            f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall_op = transform.structured.tile_using_forall %op'
            f' num_threads {threads_str}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
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
        # Parallelization should produce scf.forall
        if "scf.forall" not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.THREAD_CONFIGS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        idx = raw_slots[0] % len(cls.THREAD_CONFIGS)
        m_threads, n_threads = cls.THREAD_CONFIGS[idx]
        return {"num_threads": [m_threads, n_threads]}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        # tile_using_forall uses ceilDiv for non-divisible thread counts; no hard constraint.
        return None
