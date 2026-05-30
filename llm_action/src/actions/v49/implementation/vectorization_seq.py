import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class VectorizationSeq(ActionBase):
    """Tile to vector sizes using sequential for-loops, then vectorize.

    For conv2d ops, applies im2col lowering internally to convert
    the windowed access pattern into a vectorizable contraction.
    Single-shot: vectorization consumes the linalg op, producing vector/scf ops.
    """

    unique_execution: bool = True  # consumes the linalg op, lowers to vector + scf

    VOCAB = [1, 4, 8, 16, 32, 64]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector/tile size per loop dimension for SIMD vectorization. 1 means tile to 1 (scalar).",
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
        if any(s < 1 for s in vector_sizes):
            return False
        product = 1
        for s in vector_sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        if all(s == 1 for s in vector_sizes):
            return False
        return True

    @classmethod
    def _apply_im2col(cls, code: str) -> str:
        """Convert conv2d to im2col (matmul-like contraction) if conv2d is present."""
        transform_code = (
            'module attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n'
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            '    %img2col, %transformed = transform.structured.convert_conv2d_to_img2col %op : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            '    %matmul = transform.get_producer_of_operand %transformed[0] : (!transform.any_op) -> !transform.any_op\n'
            '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            '    transform.annotate %matmul "tag" = %tag : !transform.any_op, !transform.any_param\n'
            '    transform.yield\n'
            '  }\n'
            '}\n'
        )
        return run_transform_code(code, transform_code)

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        if "conv_2d_nchw_fchw" in code:
            try:
                code = cls._apply_im2col(code)
            except Exception:
                pass
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]

        # Conv2d requires im2col preprocessing for vectorization to work.
        if "conv_2d_nchw_fchw" in code:
            try:
                code = cls._apply_im2col(code)
            except Exception:
                return code

        n_dims = len(vector_sizes)
        n_loops = n_dims
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
        # After im2col, conv2d becomes a 4-dim contraction. Use 4 slots.
        return 4

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * 4

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(4)]
        product = 1
        for s in sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            sizes = [min(s, 4) for s in sizes]
        return {"vector_sizes": sizes}
