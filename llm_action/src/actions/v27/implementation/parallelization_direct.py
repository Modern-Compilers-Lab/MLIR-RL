from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class ParallelizationDirect(ActionBase):
    """Distribute iterations of outer parallel loops directly across threads.

    Uses tile_using_forall with num_threads to divide the iteration space
    evenly among a specified number of threads per dimension.  Thread counts
    must be powers of 2 to avoid producing dynamic shapes.  Only parallel
    dimensions should receive non-zero thread counts.
    """

    # Introduces scf.forall, fundamentally changes loop structure; one-shot.
    unique_execution: bool = True

    VOCAB = [0, 2, 4, 8, 16]  # 0 = do not parallelize that dimension

    @classmethod
    def parameters(cls) -> dict:
        return {
            "num_threads": {
                "description": "Number of threads per dimension; 0 means do not parallelize.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        num_threads = params.get("num_threads", [])
        if not num_threads or not isinstance(num_threads, list):
            return False
        # At least one non-zero thread count required
        if all(t == 0 for t in num_threads):
            return False
        if any(not isinstance(t, int) or t < 0 for t in num_threads):
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
            f'    %tiled_op, %forall = transform.structured.tile_using_forall %op num_threads {num_threads} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
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
        if 'tag = "operation_0"' not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        threads = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"num_threads": threads}
