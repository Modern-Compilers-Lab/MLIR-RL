import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class VectorizationSequential(ActionBase):
    """Lower innermost loops to SIMD vector operations with sequential tiling preprocessing.

    For conv2d, applies im2col first to eliminate windowed access patterns that prevent
    vectorization. Then tiles the contraction to vector sizes and vectorizes.

    Unique: lowering transform — the linalg op is consumed and replaced by vector ops.
    """

    unique_execution = True  # linalg op consumed, lowered to vector + scf.for

    VOCAB = [1, 2, 4]  # f64 on AVX2: 4 lanes optimal; 1 = scalar (no vectorization on that dim)

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "SIMD width per loop dimension; 1 means no vectorization on that dim",
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
        # Reject if all sizes are 1 (no actual vectorization)
        if all(s == 1 for s in vector_sizes):
            return False
        # Enforce vector product limit
        product = 1
        for s in vector_sizes:
            product *= s
        if product > 2048:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]

        # Step 1: Im2col preprocessing for conv2d (windowed patterns prevent vectorization)
        working_code = code
        is_conv2d = "linalg.conv_2d" in code
        if is_conv2d:
            img2col_transform = (
                "module attributes {transform.with_named_sequence} {\n"
                "  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n"
                '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                "    %img2col, %transformed = transform.structured.convert_conv2d_to_img2col %op"
                " : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
                "    %matmul = transform.get_producer_of_operand %transformed[0] : (!transform.any_op) -> !transform.any_op\n"
                '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
                '    transform.annotate %matmul "tag" = %tag : !transform.any_op, !transform.any_param\n'
                "    transform.yield\n"
                "  }\n"
                "}\n"
            )
            try:
                working_code = run_transform_code(code, img2col_transform)
            except Exception:
                return code
            # Post-im2col contraction has 4 dims; use first 4 vector sizes
            vs = vector_sizes[:4] if len(vector_sizes) >= 4 else vector_sizes
        else:
            vs = vector_sizes

        # Step 2: Tile ALL dims to vector sizes, then vectorize
        n_loops = len(vs)
        loop_types = ", ".join(["!transform.any_op"] * n_loops)

        vec_transform = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {str(vs)}"
            f" : (!transform.any_op) -> (!transform.any_op, {loop_types})\n"
            f"    transform.structured.vectorize %tiled_op vector_sizes {str(vs)} : !transform.any_op\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )

        try:
            return run_transform_code(working_code, vec_transform)
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
        n = min(n_loops, MAX_PARAM_SLOTS)
        return [len(cls.VOCAB)] * n

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"vector_sizes": sizes}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n = min(n_loops, MAX_PARAM_SLOTS)
        masks = []
        for i in range(n):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            slot_mask = np.array(
                [bound > 0 and bound % s == 0 for s in cls.VOCAB],
                dtype=bool,
            )
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        return np.concatenate(masks)
