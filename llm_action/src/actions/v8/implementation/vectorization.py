import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """
    Lower innermost loop dimensions to SIMD vector operations, processing
    multiple data elements per instruction using the target ISA vector width.

    Tiles the tagged operation to the requested vector sizes first (preprocessing),
    then vectorizes the tiled inner operation. This ensures vector sizes always
    match the iteration space dimensions of the inner op.
    """

    VOCAB = [1, 2, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector sizes per loop dimension. All must be positive. "
                               "The operation is first tiled to these sizes, then vectorized.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes")
        if not vector_sizes or not isinstance(vector_sizes, list):
            return False
        if not all(isinstance(s, int) and s > 0 for s in vector_sizes):
            return False
        # Reject all-ones (no-op vectorization)
        if all(s == 1 for s in vector_sizes):
            return False
        # Check vector size limit
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
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]

        # Use vector_sizes directly as tile_sizes so ALL dimensions are tiled
        # to match vector_sizes exactly in the inner op's iteration space.
        n_loops = len(vector_sizes)
        result_types = ", ".join(["!transform.any_op"] * n_loops)

        # Tile to vector sizes, then vectorize the tiled inner op
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op'
            f' tile_sizes {str(vector_sizes)}'
            f' : (!transform.any_op) -> (!transform.any_op, {result_types})\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {str(vector_sizes)}'
            f' : !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
        except Exception:
            return code

        # Validate vector sizes in the result
        if not cls._validate_vectors(result):
            return code

        return result

    @classmethod
    def _validate_vectors(cls, code: str) -> bool:
        """Check that all vector types in the result satisfy safety constraints."""
        vector_pattern = re.compile(r'vector<([^>]+)>')
        for match in vector_pattern.finditer(code):
            dims_str = match.group(1)
            parts = dims_str.split('x')
            dims = []
            for p in parts:
                p = p.strip()
                try:
                    dims.append(int(p))
                except ValueError:
                    continue
            if not dims:
                continue
            product = 1
            for d in dims:
                product *= d
            if product > VECTORIZATION_SIZE_LIMIT:
                return False
        return True

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
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS) + [1] * max(0, MAX_PARAM_SLOTS - n_loops)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        sizes += [1] * max(0, n_loops - MAX_PARAM_SLOTS)
        # Clamp total product to VECTORIZATION_SIZE_LIMIT
        product = 1
        for s in sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            for i in range(len(sizes) - 1, -1, -1):
                while sizes[i] > 1 and product > VECTORIZATION_SIZE_LIMIT:
                    sizes[i] //= 2
                    product = 1
                    for s in sizes:
                        product *= s
        return {"vector_sizes": sizes}
