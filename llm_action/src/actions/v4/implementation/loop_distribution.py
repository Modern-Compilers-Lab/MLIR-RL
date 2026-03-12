from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopDistribution(ActionBase):
    """
    Split a loop body containing the tagged operation by tiling with
    specific sizes, effectively distributing the computation across
    separate loop iterations. Uses transform.structured.split_reduction
    to split reduction dimensions into separate loops.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "split_factor": {
                "description": "Factor by which to split the reduction dimension.",
                "type": "int",
                "default": None,
            },
            "insert_split_dimension": {
                "description": "Dimension index at which to insert the split. Defaults to 0.",
                "type": "int",
                "default": 0,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        factor = params.get("split_factor")
        if not isinstance(factor, int) or factor < 2:
            return False
        dim = params.get("insert_split_dimension", 0)
        if not isinstance(dim, int) or dim < 0:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        factor = params["split_factor"]
        dim = params.get("insert_split_dimension", 0)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %init_or_alloc, %fill, %split_linalg, %combining_linalg ='
            f' transform.structured.split_reduction %op'
            f' {{split_factor = {factor}, insert_split_dimension = {dim}}}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)\n'
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
