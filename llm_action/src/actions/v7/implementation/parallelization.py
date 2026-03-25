from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Parallelization(ActionBase):
    """
    Distributes iterations of the first parallel loop dimension of a tagged
    linalg operation across multiple threads using scf.forall.

    Uses transform.structured.tile_using_forall with num_threads on the
    operation matched by tag = "operation_0". Re-annotates the inner
    tiled operation with the tag.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "num_threads": {
                "description": "Number of threads to distribute the first parallel "
                "loop dimension across.",
                "type": "int",
                "values": [2, 4, 7, 8, 14, 28],
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        n = params.get("num_threads", 0)
        if not isinstance(n, int) or n <= 0:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        n = params["num_threads"]

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main("
            f"%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} '
            f"in %arg1 : (!transform.any_op) -> !transform.any_op\n"
            f"    %tiled_op, %forall = "
            f"transform.structured.tile_using_forall %op "
            f"num_threads [{n}] "
            f": (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f"    transform.annotate %tiled_op "
            f'"tag" = %tag : !transform.any_op, !transform.any_param\n'
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
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
