from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class BufferizationStrategy(ActionBase):
    """
    Choose how tensor-semantic operands of the target linalg op are converted
    into buffer-semantic operands, affecting in-place updates, allocation
    placement, and the legality of later fusion/promotion steps.

    Supported strategies:
      - strategy = 0 ("eliminate"): apply `eliminate_empty_tensors` and
        `empty_tensor_to_alloc_tensor` as a preparation pass before the
        downstream one-shot bufferize. This normalizes allocation points in
        the tensor-level IR without introducing any new memref ops.
      - strategy = 1 ("alloc-destination"): eagerly materialize the
        destination operand of the target op into a fresh local allocation
        using `transform.structured.bufferize_to_allocation
        {bufferize_destination_only}`. This guarantees a private writable
        buffer for the result and decouples the op from the function
        boundary's original destination tensor.
    """

    STRATEGY_VOCAB = [0, 1]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "strategy": {
                "description": (
                    "0 = eliminate empty tensors (tensor-level preparation); "
                    "1 = alloc-destination (bufferize the target op's "
                    "destination operand into a fresh allocation)."
                ),
                "type": "int",
                "values": cls.STRATEGY_VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        strategy = params.get("strategy")
        if not isinstance(strategy, int) or strategy not in (0, 1):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        strategy = int(params["strategy"])
        if strategy == 0:
            transform_code = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    transform.structured.eliminate_empty_tensors %arg1 : !transform.any_op
    %empty = transform.structured.match ops{["tensor.empty"]} in %arg1 : (!transform.any_op) -> !transform.op<"tensor.empty">
    transform.bufferization.empty_tensor_to_alloc_tensor %empty : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
    transform.yield
  }
}
"""
        else:
            transform_code = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op
    %buf, %new_ops = transform.structured.bufferize_to_allocation %op {bufferize_destination_only} : !transform.any_op
    %matched = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tag = transform.param.constant "operation_0" -> !transform.any_param
    transform.annotate %matched "tag" = %tag : !transform.any_op, !transform.any_param
    transform.yield
  }
}
"""
        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if not after or "func.func" not in after:
            return False
        if 'tag = "operation_0"' not in after:
            return False
        strategy = int(params.get("strategy", 0))
        if strategy == 0:
            # Preparation pass may be a no-op on already-clean IR.
            return True
        # alloc-destination must introduce a fresh allocation.
        if after.strip() == before.strip():
            return False
        return "memref.alloc" in after or "bufferization.to_tensor" in after

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.STRATEGY_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        strategy = cls.STRATEGY_VOCAB[raw_slots[0] % len(cls.STRATEGY_VOCAB)]
        return {"strategy": strategy}
