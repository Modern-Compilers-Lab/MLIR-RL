import re
from functools import reduce
from operator import mul

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Vectorize a linalg operation after tiling to specified vector sizes.

    One-shot: consumes the linalg op and produces vector.* + scf.for ops.
    For conv2d ops, automatically applies Im2col lowering first to convert
    windowed access patterns into vectorizable projected-permutation maps.
    """

    unique_execution: bool = True  # linalg op is consumed, replaced by vector ops

    VOCAB = [1, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector size per loop dimension for tiling+vectorization.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def _detect_n_loops(cls, code: str) -> int:
        """Detect the number of loops in the tagged linalg operation."""
        # Find iterator_types associated with the tagged op
        for match in re.finditer(r'iterator_types\s*=\s*\[([^\]]+)\]', code):
            start = match.start()
            # Check if tag = "operation_0" appears in the same op (within ~500 chars)
            window = code[start:start + 500]
            if 'tag = "operation_0"' in window:
                types_str = match.group(1)
                return len(re.findall(r'"(?:parallel|reduction)"', types_str))
        # Fallback for named ops without explicit iterator_types in text
        if "linalg.conv_2d_nchw_fchw" in code:
            return 7
        if "linalg.matmul" in code:
            return 3
        return 0

    @classmethod
    def _apply_im2col(cls, code: str) -> str:
        """Apply Im2col lowering to convert conv2d to matmul-like generic."""
        from llm_action.src.actions.v28.implementation.im2col_lowering import Im2colLowering
        if Im2colLowering.precondition(code, {}):
            result = Im2colLowering.implement(code, {})
            if Im2colLowering.postcondition(code, result, {}):
                return result
        return code

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes", [])
        if not vector_sizes or not isinstance(vector_sizes, list):
            return False
        if all(s <= 1 for s in vector_sizes):
            return False
        if "linalg." not in code:
            return False
        # For conv code, im2col preprocessing reduces to 4 loops:
        # [batch, filter, spatial, reduction]. Only parallel dims (first 3)
        # determine the output vector size (register pressure). The reduction
        # dim is handled via accumulation and doesn't bloat registers.
        if "linalg.conv_" in code:
            effective_sizes = vector_sizes[:4]
            parallel_sizes = [s for s in effective_sizes[:3] if s > 1]
            if not parallel_sizes:
                return False
            product = reduce(mul, parallel_sizes, 1)
        else:
            product = reduce(mul, vector_sizes, 1)
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """If code contains a conv op, apply Im2col to make it vectorizable."""
        if "linalg.conv_" not in code:
            return code
        return cls._apply_im2col(code)

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        original_code = code
        # Auto-preprocess conv code (handles MCP path that skips preprocess)
        if "linalg.conv_" in code:
            code = cls._apply_im2col(code)
            if "linalg.conv_" in code:
                # Im2col failed, cannot vectorize
                return original_code

        vector_sizes = params["vector_sizes"]

        # Detect actual loop count in the tagged op and adapt vector_sizes
        n_loops = cls._detect_n_loops(code)
        if n_loops > 0 and len(vector_sizes) > n_loops:
            vector_sizes = vector_sizes[:n_loops]
        elif n_loops > 0 and len(vector_sizes) < n_loops:
            # Pad with 1s for missing dimensions
            vector_sizes = list(vector_sizes) + [1] * (n_loops - len(vector_sizes))
        else:
            vector_sizes = list(vector_sizes)

        # Tile sizes = vector sizes for dimensions we want to vectorize
        tile_sizes = list(vector_sizes)
        n_tiled = sum(1 for s in tile_sizes if s > 1)

        if n_tiled == 0:
            return original_code

        vector_sizes_str = "[" + ", ".join(str(s) for s in vector_sizes) + "]"

        # Only tile dims with size > 1, rest set to 0
        effective_tiles = [s if s > 1 else 0 for s in tile_sizes]
        n_effective = sum(1 for s in effective_tiles if s != 0)

        if n_effective == 0:
            # No tiling needed, vectorize directly
            transform_code = (
                f'module attributes {{transform.with_named_sequence}} {{\n'
                f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
                f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f'    transform.structured.vectorize %op vector_sizes {vector_sizes_str} : !transform.any_op\n'
                f'    transform.yield\n'
                f'  }}\n'
                f'}}\n'
            )
        else:
            loop_results = ", ".join(["!transform.any_op"] * n_effective)
            effective_tiles_str = "[" + ", ".join(str(s) for s in effective_tiles) + "]"

            transform_code = (
                f'module attributes {{transform.with_named_sequence}} {{\n'
                f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
                f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f'    %tiled_op, %loops:{n_effective} = transform.structured.tile_using_for %op tile_sizes {effective_tiles_str}'
                f' : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
                f'    transform.structured.vectorize %tiled_op vector_sizes {vector_sizes_str} : !transform.any_op\n'
                f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
                f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
                f'    transform.yield\n'
                f'  }}\n'
                f'}}\n'
            )

        try:
            result = run_transform_code(code, transform_code)
            # Validate: only reject if the WRITE vectors (output, excluding
            # intermediates used for multi_reduction) exceed the limit.
            # Intermediate vectors naturally include the reduction dimension
            # and will be larger; they are lowered away by the pass pipeline.
            write_vectors = re.findall(
                r'vector\.transfer_write\s+\S+\s*,\s*\S+\[.*?\]\s*:\s*(vector<[^>]+>)',
                result,
            )
            for vec_type in write_vectors:
                try:
                    dims_str = vec_type[len("vector<"):-1]  # strip "vector<" and ">"
                    shape_parts = re.sub(r'x[a-z]\w*$', '', dims_str).split('x')
                    numeric_dims = [int(d) for d in shape_parts if d.strip().isdigit()]
                    if numeric_dims:
                        prod = reduce(mul, numeric_dims, 1)
                        if prod > VECTORIZATION_SIZE_LIMIT:
                            return original_code
                except (ValueError, IndexError):
                    pass
            return result
        except Exception:
            return original_code

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
        return {"vector_sizes": sizes}
