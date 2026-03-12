from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class ParallelizationAction(ActionBase):
    """
    Parallelization Action: Tiles the tagged operation using scf.forall,
    distributing iterations across threads for parallel execution.

    Uses transform.structured.tile_using_forall with num_threads to create
    a parallel loop nest.

    Parameters:
        num_threads (list[int]): Number of threads per loop dimension.
            0 means "do not parallelize that dimension".
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "num_threads": {
                "description": "Number of threads for each loop dimension. 0 means do not parallelize.",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False

        num_threads = params.get("num_threads", None)
        if num_threads is None or not isinstance(num_threads, list):
            return False

        if len(num_threads) == 0:
            return False

        if not all(isinstance(t, int) and t >= 0 for t in num_threads):
            return False

        # At least one dimension must be parallelized
        if all(t == 0 for t in num_threads):
            return False

        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        num_threads = params["num_threads"]
        threads_str = "[" + ", ".join(str(t) for t in num_threads) + "]"

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall_op = transform.structured.tile_using_forall %op'
            f' num_threads {threads_str}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
            return result
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        if len(after.strip()) == 0:
            return False
        return True
