from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopCoalescing(ActionBase):
    """
    Merge multiple nested loops with independent iteration ranges into
    a single flat loop, simplifying loop control and enabling uniform
    work distribution. Gets the outermost parent loop of the tagged
    operation and coalesces the nested loop structure.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "loop_depth": {
                "description": "Depth of the outermost loop to start coalescing from "
                               "(relative to the tagged op). 1 = immediate parent.",
                "type": "int",
                "default": 1,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        depth = params.get("loop_depth", 1)
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
        depth = params.get("loop_depth", 1)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %parent = transform.get_parent_op %op {{op_name = "scf.for", nth_parent = {depth}}}'
            f' : (!transform.any_op) -> !transform.op<"scf.for">\n'
            f'    %coalesced = transform.loop.coalesce %parent'
            f' : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)\n'
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
