from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Parallelization(ActionBase):
    """
    Parallelization action: distributes iterations of a linalg operation across
    multiple threads by tiling into scf.forall using transform.structured.tile_using_forall.
    This generates parallel loop nests that can be lowered to OpenMP.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "num_threads": {
                "description": "List of thread counts per dimension. Each entry specifies how many threads to use for that dimension. 0 means no parallelization along that dimension.",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        num_threads = params.get("num_threads")
        if not num_threads or not isinstance(num_threads, list):
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

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall_op = transform.structured.tile_using_forall %op'
            f' num_threads {num_threads}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
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
        if "scf.forall" not in after:
            return False
        return True
