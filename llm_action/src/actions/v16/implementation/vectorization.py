from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """
    Vectorization action: maps iterations to SIMD vector lanes using
    transform.structured.vectorize. Preprocessing generalizes named ops
    and tiles to vector sizes.
    For conv2d ops, preprocessing applies im2col conversion first since
    vectorization cannot handle the sliding-window affine maps directly.
    """

    VOCAB = [0, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "Vector sizes per dimension (0 = do not vectorize that dim)",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes", [])
        if not vector_sizes or all(s == 0 for s in vector_sizes):
            return False
        # Check vector size product doesn't exceed limit
        product = 1
        for s in vector_sizes:
            if s > 0:
                product *= s
        if product > 1024:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]

        is_conv2d = "conv_2d" in code and 'tag = "operation_0"' in code

        if is_conv2d:
            return cls._implement_conv2d(code, vector_sizes)
        else:
            return cls._implement_generic(code, vector_sizes)

    @classmethod
    def _implement_conv2d(cls, code: str, vector_sizes: list[int]) -> str:
        """For conv2d: im2col → tile → vectorize."""
        # After im2col, the matmul-like generic has 4 iterators: batch, F, OH*OW, C*KH*KW
        # We must tile ALL 4 dims; user vector_sizes fill dims 1-3, dim 0 (batch) is always 1
        vs = [s for s in vector_sizes if s != 0]
        if not vs:
            return code

        # Build 4-dim tile/vector sizes: [1, vs[0], vs[1], vs[2]] (pad with 1s)
        padded = list(vs)
        while len(padded) < 3:
            padded.append(1)
        full_vs = [1] + padded[:3]  # Always 4 dims, all non-zero

        loop_handles = ", ".join([f"%l{i}" for i in range(4)])
        r = ", ".join(["!transform.any_op"] * 5)

        vec_str = str(full_vs)

        transform_code = (
            f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %img2col, %transformed = transform.structured.convert_conv2d_to_img2col %op'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %matmul = transform.get_producer_of_operand %transformed[0]'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %matmul "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    %tiled_matmul, {loop_handles} = transform.structured.tile_using_for %matmul'
            f' tile_sizes {str(full_vs)} : (!transform.any_op) -> ({r})\n'
            f'    transform.structured.vectorize %tiled_matmul vector_sizes {vec_str}'
            f' : !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
            # Validate vector sizes in output
            if not cls._validate_vectors(result):
                return code
            return result
        except Exception:
            return code

    @classmethod
    def _implement_generic(cls, code: str, vector_sizes: list[int]) -> str:
        """For generic/matmul ops: generalize → tile → vectorize."""
        n_iterators = cls._count_iterators(code)
        full_vs = list(vector_sizes) + [0] * max(0, n_iterators - len(vector_sizes))
        full_vs = full_vs[:n_iterators]

        n_nonzero = sum(1 for s in full_vs if s != 0)
        if n_nonzero == 0:
            return code

        loop_handles = ", ".join([f"%l{i}" for i in range(n_nonzero)])
        r = ", ".join(["!transform.any_op"] * (1 + n_nonzero))

        # Vector sizes for vectorize: replace 0s with 1s
        vec_sizes = [s if s != 0 else 1 for s in full_vs]
        vec_str = str(vec_sizes)

        # Check if this is a named op that needs generalization
        needs_generalize = any(
            named in code
            for named in ["linalg.matmul", "linalg.conv", "linalg.batch_matmul"]
        )

        generalize_line = ""
        op_handle = "%op"
        if needs_generalize:
            generalize_line = (
                f'    %generic = transform.structured.generalize %op'
                f' : (!transform.any_op) -> !transform.any_op\n'
            )
            op_handle = "%generic"

        transform_code = (
            f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'{generalize_line}'
            f'    %tiled_op, {loop_handles} = transform.structured.tile_using_for {op_handle}'
            f' tile_sizes {str(full_vs)} : (!transform.any_op) -> ({r})\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.structured.vectorize %tiled_op vector_sizes {vec_str}'
            f' : !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
            if not cls._validate_vectors(result):
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
        # Clamp product to <= 1024
        product = 1
        for s in sizes:
            if s > 0:
                product *= s
        if product > 1024:
            for i in range(len(sizes) - 1, -1, -1):
                if sizes[i] > 2:
                    sizes[i] = 2
                    product = 1
                    for s in sizes:
                        if s > 0:
                            product *= s
                    if product <= 1024:
                        break
        return {"vector_sizes": sizes}

    @staticmethod
    def _count_iterators(code: str) -> int:
        import re
        match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
        if match:
            return len(match.group(1).split(","))
        if "conv_2d_nchw_fchw" in code:
            return 7
        if "matmul" in code:
            return 3
        return 7

    @staticmethod
    def _validate_vectors(code: str) -> bool:
        """Validate that vector types in the output are within safe bounds."""
        import re
        for match in re.finditer(r'vector<([^>]+)>', code):
            dims_str = match.group(1)
            parts = dims_str.split('x')
            dims = [p for p in parts if p.strip().isdigit()]
            if dims:
                product = 1
                for d in dims:
                    product *= int(d)
                if product > 1024:
                    return False
        return True
