from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopUnrolling(ActionBase):
    """
    Replicate the loop body multiple times per iteration, reducing branch
    overhead and exposing independent instructions for pipelining.
    Gets the parent scf.for loop of the tagged operation and unrolls it.
    The nth_parent_op parameter controls which enclosing loop to target.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "unroll_factor": {
                "description": "Number of loop body replications.",
                "type": "int",
                "default": None,
            },
            "loop_depth": {
                "description": "Which enclosing loop to unroll (1 = innermost parent, 2 = next outer, etc.).",
                "type": "int",
                "default": 1,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        factor = params.get("unroll_factor")
        if not isinstance(factor, int) or factor < 2:
            return False
        depth = params.get("loop_depth", 1)
        if not isinstance(depth, int) or depth < 1:
            return False
        # Must have scf.for loops around the tagged op
        if "scf.for" not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        factor = params["unroll_factor"]
        depth = params.get("loop_depth", 1)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %parent = transform.get_parent_op %op {{op_name = "scf.for", nth_parent = {depth}}}'
            f' : (!transform.any_op) -> !transform.op<"scf.for">\n'
            f'    transform.loop.unroll %parent {{factor = {factor}}}'
            f' : !transform.op<"scf.for">\n'
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
