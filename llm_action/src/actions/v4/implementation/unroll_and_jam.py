from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class UnrollAndJam(ActionBase):
    """
    Unroll an outer loop by a given factor and fuse (jam) the replicated
    inner loop bodies, creating multiple independent computation streams
    in the innermost loop. This is implemented by getting the outer parent
    loop of the tagged operation and unrolling it.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "jam_factor": {
                "description": "Number of copies of the inner body to fuse.",
                "type": "int",
                "default": None,
            },
            "outer_loop_depth": {
                "description": "Which outer loop to unroll (1 = immediate parent, "
                               "2 = grandparent, etc.). Should target an outer loop "
                               "relative to the tagged op.",
                "type": "int",
                "default": 2,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        factor = params.get("jam_factor")
        if not isinstance(factor, int) or factor < 2:
            return False
        depth = params.get("outer_loop_depth", 2)
        if not isinstance(depth, int) or depth < 1:
            return False
        if "scf.for" not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        factor = params["jam_factor"]
        depth = params.get("outer_loop_depth", 2)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %outer = transform.get_parent_op %op {{op_name = "scf.for", nth_parent = {depth}}}'
            f' : (!transform.any_op) -> !transform.op<"scf.for">\n'
            f'    transform.loop.unroll %outer {{factor = {factor}}}'
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
