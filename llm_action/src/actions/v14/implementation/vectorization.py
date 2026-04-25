import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import (
    MAX_PARAM_SLOTS,
    MAX_VOCAB_SIZE_PER_SLOT,
    VECTORIZATION_SIZE_LIMIT,
)
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """
    Vectorize the innermost structured loop band of the target linalg op using
    `transform.structured.vectorize`. The target is first tiled with
    `vector_sizes` so that the resulting static vectors are hardware-realistic
    (AVX2-friendly) and bounded in total size.

    Parameters:
      - vector_sizes: list[int], per-loop static vector widths.
    """

    VOCAB = [1, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": (
                    "Per-loop static vector widths applied to the tiled inner "
                    "band. The total product of sizes is bounded to keep "
                    "vectors within hardware-realistic limits."
                ),
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @staticmethod
    def _vector_product(sizes: list[int]) -> int:
        prod = 1
        for s in sizes:
            if s > 0:
                prod *= s
        return prod

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        sizes = params.get("vector_sizes")
        if not isinstance(sizes, (list, tuple)) or len(sizes) == 0:
            return False
        if not all(isinstance(s, int) and s >= 1 for s in sizes):
            return False
        if all(s == 1 for s in sizes):
            return False
        if cls._vector_product(list(sizes)) > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        sizes = list(params["vector_sizes"])
        n_loops = len(sizes)
        result_types = ", ".join(["!transform.any_op"] * n_loops)
        sizes_str = "[" + ", ".join(str(int(s)) for s in sizes) + "]"

        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op
    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %generic tile_sizes {sizes_str} : (!transform.any_op) -> (!transform.any_op, {result_types})
    %tag = transform.param.constant "operation_0" -> !transform.any_param
    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param
    %func = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }}
}}
"""
        try:
            transformed = run_transform_code(code, transform_code)
        except Exception:
            return code
        # Safety check: reject if any large static vector was materialized.
        if not cls._vector_size_is_safe(transformed):
            return code
        return transformed

    @staticmethod
    def _vector_size_is_safe(code: str) -> bool:
        vec_re = re.compile(r"vector<([0-9x]+)xf[0-9]+>")
        for match in vec_re.finditer(code):
            dims_str = match.group(1)
            try:
                dims = [int(d) for d in dims_str.split("x") if d]
            except ValueError:
                continue
            if len(dims) >= 3 and any(d > 16 for d in dims):
                return False
            prod = 1
            for d in dims:
                prod *= d
            if prod > VECTORIZATION_SIZE_LIMIT:
                return False
        return True

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if not after or "func.func" not in after:
            return False
        if after.strip() == before.strip():
            return False
        if 'tag = "operation_0"' not in after:
            return False
        return "vector.transfer_read" in after or "vector.transfer_write" in after

    @classmethod
    def params_size(cls) -> int:
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(max(n_loops, 1), MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(max(n_loops, 1), MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        # Clamp to stay within vector size limit.
        prod = 1
        for i in range(len(sizes)):
            if sizes[i] * prod > VECTORIZATION_SIZE_LIMIT:
                sizes[i] = 1
            prod *= sizes[i]
        return {"vector_sizes": sizes}
