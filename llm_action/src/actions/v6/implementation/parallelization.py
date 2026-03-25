from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Parallelization(ActionBase):
    """
    Distribute iterations of parallel loops across hardware threads using
    scf.forall, enabling concurrent execution on multiple CPU cores.
    Uses transform.structured.tile_using_forall with parameterized num_threads.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "num_threads": {
                "description": "List of thread counts, one per loop dimension. 0 means do not parallelize that dimension.",
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
        if not all(isinstance(n, int) and n >= 0 for n in num_threads):
            return False
        if all(n == 0 for n in num_threads):
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
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall_op = transform.structured.tile_using_forall %op'
            f' num_threads {threads_str} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return True
