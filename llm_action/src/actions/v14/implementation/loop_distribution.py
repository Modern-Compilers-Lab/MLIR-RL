from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopDistribution(ActionBase):
    """
    Distribute the iteration space of the target linalg op across two
    complementary structured ops by splitting along a chosen dimension using
    `transform.structured.split`. The resulting low/high halves are
    independent linalg ops that can be further tiled, vectorized, or
    parallelized on their own.

    Parameters:
      - dimension: int, the iteration-space dimension along which to split.
      - chunk_size: int, the size of the lower chunk of the split (must be
        strictly smaller than the extent of the selected dimension for the
        split to produce two distinct halves).
    """

    DIM_VOCAB = [0, 1, 2, 3, 4]
    CHUNK_VOCAB = [8, 16, 32, 64, 128]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "dimension": {
                "description": (
                    "The iteration-space dimension along which to split "
                    "the target linalg op."
                ),
                "type": "int",
                "values": cls.DIM_VOCAB,
            },
            "chunk_size": {
                "description": (
                    "The size of the lower chunk of the split. Must be "
                    "strictly smaller than the extent of the selected "
                    "dimension for the split to produce two parts."
                ),
                "type": "int",
                "values": cls.CHUNK_VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        dimension = params.get("dimension")
        chunk = params.get("chunk_size")
        if not isinstance(dimension, int) or dimension < 0:
            return False
        if not isinstance(chunk, int) or chunk <= 0:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        dimension = int(params["dimension"])
        chunk = int(params["chunk_size"])

        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    %split_list = transform.structured.split %op after {chunk} {{ dimension = {dimension} }} : !transform.any_op
    %low, %high = transform.split_handle %split_list : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tag = transform.param.constant "operation_0" -> !transform.any_param
    transform.annotate %low "tag" = %tag : !transform.any_op, !transform.any_param
    transform.annotate %high "tag" = %tag : !transform.any_op, !transform.any_param
    transform.yield
  }}
}}
"""
        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if not after or "func.func" not in after:
            return False
        if after.strip() == before.strip():
            return False
        if 'tag = "operation_0"' not in after:
            return False
        # Splitting should double the count of tagged linalg ops.
        return after.count("linalg") > before.count("linalg")

    @classmethod
    def params_size(cls) -> int:
        return 2

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.DIM_VOCAB), len(cls.CHUNK_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        dim_idx = raw_slots[0] % len(cls.DIM_VOCAB)
        chunk_idx = raw_slots[1] % len(cls.CHUNK_VOCAB)
        dimension = cls.DIM_VOCAB[dim_idx] % max(int(n_loops), 1)
        chunk_size = cls.CHUNK_VOCAB[chunk_idx]
        return {"dimension": dimension, "chunk_size": chunk_size}
