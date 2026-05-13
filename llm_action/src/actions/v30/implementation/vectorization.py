import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Map iterations of innermost loops to SIMD vector lanes, converting
    scalar multiply-accumulate operations into vector FMA instructions.

    For conv2d, preprocessing converts to img2col (matmul-like contraction)
    before tiling and vectorizing, since the windowed access patterns in
    conv2d prevent direct vectorization.

    Single-shot: vectorization consumes the linalg op and lowers to
    vector.* + scf.for; a second application has no valid target."""

    unique_execution: bool = True  # lowering transform; linalg op is consumed

    VOCAB = [1, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector size per loop dimension; product must not exceed SIMD capacity",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes", [])
        if not vector_sizes:
            return False
        if all(s == 1 for s in vector_sizes):
            return False
        product = 1
        for s in vector_sizes:
            if s < 1:
                return False
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def _is_conv2d(cls, code: str) -> bool:
        """Check if the tagged operation is a conv2d op."""
        return "linalg.conv_2d_nchw_fchw" in code and 'tag = "operation_0"' in code

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def _build_conv2d_vectorize_transform(cls, vector_sizes: list[int]) -> str:
        """Build transform for conv2d: img2col + tile + vectorize."""
        # After img2col, contraction has 4 dims: [N, F, OH*OW, C*KH*KW]
        # Use first 4 vector_sizes entries (pad with 1 if fewer)
        vs = list(vector_sizes[:4])
        while len(vs) < 4:
            vs.append(1)

        n_loops = sum(1 for s in vs if s != 0 and s != 1)
        # Tile all 4 dims then vectorize
        n_tile_loops = sum(1 for s in vs if s > 1)

        if n_tile_loops == 0:
            # All sizes are 1 — vectorize with sizes [1,1,1,1] directly
            # Still need tiling to create a point tile
            vs_for_tile = [1, 1, 1, 1]
        else:
            vs_for_tile = vs

        loop_results = ", ".join(["!transform.any_op"] * 4)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %img2col, %transformed = transform.structured.convert_conv2d_to_img2col %op : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %contraction = transform.get_producer_of_operand %transformed[0] : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:4 = transform.structured.tile_using_for %contraction tile_sizes {vs_for_tile} : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {vs_for_tile} : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )
        return transform_code

    @classmethod
    def _build_generic_vectorize_transform(cls, vector_sizes: list[int]) -> str:
        """Build transform for generic linalg ops: tile + vectorize."""
        n_loops = len(vector_sizes)
        loop_results = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {vector_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {vector_sizes} : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )
        return transform_code

    @classmethod
    def _check_vectors(cls, code: str) -> bool:
        """Check that generated vectors don't violate size/rank constraints."""
        vector_pattern = re.compile(r'vector<([^>]+)>')
        for match in vector_pattern.finditer(code):
            dims_str = match.group(1)
            # Extract numeric dimensions (ignore type suffix like xf64)
            parts = dims_str.split("x")
            dims = []
            for p in parts:
                p = p.strip()
                if p.isdigit():
                    dims.append(int(p))
            if not dims:
                continue
            product = 1
            for d in dims:
                product *= d
            if product > VECTORIZATION_SIZE_LIMIT:
                return False
            if len(dims) >= 3 and product > 64:
                return False
        return True

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]

        if cls._is_conv2d(code):
            transform_code = cls._build_conv2d_vectorize_transform(vector_sizes)
        else:
            transform_code = cls._build_generic_vectorize_transform(vector_sizes)

        try:
            result = run_transform_code(code, transform_code)
            if not cls._check_vectors(result):
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
        if not cls._check_vectors(after):
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
        # Clamp total product to VECTORIZATION_SIZE_LIMIT
        product = 1
        for s in sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            # Scale down: keep only the first few non-1 entries
            for i in range(len(sizes) - 1, -1, -1):
                if product <= VECTORIZATION_SIZE_LIMIT:
                    break
                if sizes[i] > 1:
                    product //= sizes[i]
                    sizes[i] = 1
        return {"vector_sizes": sizes}
