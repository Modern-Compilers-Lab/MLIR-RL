import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Map innermost loop computation to SIMD vector instructions,
    processing multiple data elements per cycle. Internally tiles to
    vector_sizes before vectorizing (self-contained action)."""

    VOCAB = [2, 4, 8, 16, 32]
    TAG = 'tag = "operation_0"'

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector sizes per loop dimension (all positive, product <= 1024)",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if cls.TAG not in code:
            return False
        vector_sizes = params.get("vector_sizes", [])
        if not vector_sizes or any(s <= 0 for s in vector_sizes):
            return False
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
        n = len(vector_sizes)

        loop_vars = ', '.join([f'%loop_{i}' for i in range(n)])
        result_types = ', '.join(['!transform.any_op'] * (n + 1))

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, {loop_vars} = transform.structured.tile_using_for %op tile_sizes {vector_sizes} : (!transform.any_op) -> ({result_types})\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {vector_sizes} : !transform.any_op\n'
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
        if 'func.func' not in after:
            return False
        # Vector safety check: ensure no oversized vectors
        for match in re.finditer(r'vector<([^>]+)>', after):
            type_str = match.group(1)
            parts = type_str.split('x')
            dims = []
            for p in parts:
                try:
                    dims.append(int(p))
                except ValueError:
                    pass
            if dims:
                product = 1
                for d in dims:
                    product *= d
                if product > VECTORIZATION_SIZE_LIMIT:
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
        # Pad for extra dimensions beyond MAX_PARAM_SLOTS
        while len(sizes) < n_loops:
            sizes.append(cls.VOCAB[0])
        # Clamp product to VECTORIZATION_SIZE_LIMIT
        product = 1
        for s in sizes:
            product *= s
        while product > VECTORIZATION_SIZE_LIMIT:
            reduced = False
            for i in range(len(sizes) - 1, -1, -1):
                if sizes[i] > cls.VOCAB[0]:
                    curr_idx = cls.VOCAB.index(sizes[i])
                    sizes[i] = cls.VOCAB[curr_idx - 1]
                    reduced = True
                    break
            if not reduced:
                break
            product = 1
            for s in sizes:
                product *= s
        return {"vector_sizes": sizes}
