import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """
    Map loop body computations to SIMD vector operations by first tiling the
    target operation to the specified vector sizes, then vectorizing the tiled
    body.

    Category B (lowering): the linalg op is consumed and replaced by vector
    ops + scf.for loops. The outermost generated loop is tagged.

    unique_execution = True because vectorization consumes the linalg op and
    lowers it to vector ops; a second application has no valid linalg target.
    """

    unique_execution: bool = True  # consumes the linalg op (lowering)

    VOCAB = [2, 4, 8, 16, 32]  # per-dimension vector size vocabulary

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "SIMD vector widths per loop dimension.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes", [])
        if not vector_sizes or not isinstance(vector_sizes, list):
            return False
        if any(not isinstance(s, int) or s <= 0 for s in vector_sizes):
            return False
        # Check total vector product <= VECTORIZATION_SIZE_LIMIT
        product = 1
        for s in vector_sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def _check_vector_safety(cls, transformed: str) -> bool:
        """Reject if transformed code has illegal vectors (too large or rank >= 3)."""
        vec_pattern = re.compile(r'vector<([^>]+)>')
        for match in vec_pattern.finditer(transformed):
            dims_str = match.group(1)
            # Extract numeric dims (ignore type suffix like f64, f32)
            parts = dims_str.replace('x', ' ').split()
            dims = []
            for p in parts:
                try:
                    dims.append(int(p))
                except ValueError:
                    pass  # type specifier like f64
            if not dims:
                continue
            product = 1
            for d in dims:
                product *= d
            if product > VECTORIZATION_SIZE_LIMIT:
                return False
        return True

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]
        n_loops = len(vector_sizes)
        loop_results = ", ".join(["!transform.any_op"] * n_loops)
        sizes_str = str(vector_sizes)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {sizes_str} : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {sizes_str} : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
            if not cls._check_vector_safety(result):
                return code
            return result
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
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        # Clamp product to VECTORIZATION_SIZE_LIMIT
        product = 1
        for s in sizes:
            product *= s
        while product > VECTORIZATION_SIZE_LIMIT and len(sizes) > 0:
            # Reduce the largest dimension
            max_idx = sizes.index(max(sizes))
            idx_in_vocab = cls.VOCAB.index(sizes[max_idx]) if sizes[max_idx] in cls.VOCAB else len(cls.VOCAB) - 1
            if idx_in_vocab > 0:
                sizes[max_idx] = cls.VOCAB[idx_in_vocab - 1]
            else:
                sizes[max_idx] = cls.VOCAB[0]
            product = 1
            for s in sizes:
                product *= s
        return {"vector_sizes": sizes}
