from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class BufferizationStrategy(ActionBase):
    """
    Select the strategy for converting tensor-semantic operations into
    buffer-semantic operations:
    - Strategy 0: Eliminate empty tensors before bufferization.
    - Strategy 1: Convert empty tensors to explicit alloc_tensors.
    """

    STRATEGY_OPTIONS = [0, 1]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "strategy": {
                "description": "Bufferization strategy. 0=elimination, 1=explicit allocation.",
                "type": "int",
                "values": cls.STRATEGY_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        strategy = params.get("strategy", -1)
        if strategy not in (0, 1):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        strategy = params["strategy"]

        if strategy == 0:
            # Elimination-based: eliminate empty tensors, then bufferize
            transform_code = (
                f'module attributes {{transform.with_named_sequence}} {{\n'
                f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.consumed}}) {{\n'
                f'    transform.structured.eliminate_empty_tensors %arg1 : !transform.any_op\n'
                f'    %empty = transform.structured.match ops{{["tensor.empty"]}} in %arg1'
                f' : (!transform.any_op) -> !transform.op<"tensor.empty">\n'
                f'    transform.bufferization.empty_tensor_to_alloc_tensor %empty'
                f' : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">\n'
                f'    %arg2 = transform.bufferization.one_shot_bufferize'
                f' layout{{IdentityLayoutMap}} %arg1'
                f' {{bufferize_function_boundaries = true}}'
                f' : (!transform.any_op) -> !transform.any_op\n'
                f'    transform.yield\n'
                f'  }}\n'
                f'}}\n'
            )
        else:
            # Explicit allocation: directly bufferize without elimination
            transform_code = (
                f'module attributes {{transform.with_named_sequence}} {{\n'
                f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.consumed}}) {{\n'
                f'    %arg2 = transform.bufferization.one_shot_bufferize'
                f' layout{{IdentityLayoutMap}} %arg1'
                f' {{bufferize_function_boundaries = true}}'
                f' : (!transform.any_op) -> !transform.any_op\n'
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

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.STRATEGY_OPTIONS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {"strategy": cls.STRATEGY_OPTIONS[raw_slots[0] % len(cls.STRATEGY_OPTIONS)]}
