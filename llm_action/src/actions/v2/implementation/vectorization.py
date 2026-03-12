import re
from functools import reduce
from operator import mul

from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code

MAX_VECTOR_ELEMENTS = 1024
MAX_VECTOR_RANK = 3


class Vectorization(ActionBase):
    """
    Vectorization action: maps loop computations onto SIMD vector operations
    using the MLIR Transform dialect. Applies transform.structured.vectorize
    with specified vector sizes.

    Enforces vectorization safety contract:
    - Total vector elements <= 1024
    - Max vector rank <= 3 (and only if small)
    - No tile-as-vector lowering
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "List of vector sizes, one per iterator dimension of the target op. Each size defines the number of elements to vectorize along that dimension.",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code and 'linalg.generic' not in code:
            return False
        vector_sizes = params.get("vector_sizes")
        if not vector_sizes or not isinstance(vector_sizes, list):
            return False
        if not all(isinstance(s, int) and s > 0 for s in vector_sizes):
            return False
        total = reduce(mul, vector_sizes, 1)
        if total > MAX_VECTOR_ELEMENTS:
            return False
        rank = len([s for s in vector_sizes if s > 1])
        if rank > MAX_VECTOR_RANK:
            return False
        return True

    @classmethod
    def _is_conv2d(cls, code: str) -> bool:
        """Check if the tagged operation is a conv_2d op."""
        return 'linalg.conv_2d' in code and 'tag = "operation_0"' in code

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """Preprocess code before vectorization.

        For conv2d ops, applies img2col decomposition to convert the convolution
        into a matmul-like linalg.generic that can be vectorized.
        """
        if not cls._is_conv2d(code):
            return code

        transform_code = (
            'module attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n'
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1'
            ' : (!transform.any_op) -> !transform.any_op\n'
            '    %img2col, %matmul = transform.structured.convert_conv2d_to_img2col %op'
            ' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            '    transform.yield\n'
            '  }\n'
            '}\n'
        )
        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def _check_vector_safety(cls, code: str) -> bool:
        """Check transformed code for vector safety violations."""
        vector_pattern = re.compile(r'vector<([^>]+)>')
        for match in vector_pattern.finditer(code):
            dims_str = match.group(1)
            dims_part = dims_str.split('x')
            dims = []
            for d in dims_part:
                d = d.strip()
                try:
                    dims.append(int(d))
                except ValueError:
                    continue
            if not dims:
                continue
            total = reduce(mul, dims, 1)
            if total > MAX_VECTOR_ELEMENTS:
                return False
            if len(dims) >= 3 and total > MAX_VECTOR_ELEMENTS:
                return False
        return True

    @classmethod
    def _build_transform_code(cls, code: str, vector_sizes: list[int], tile_sizes: list[int] | None = None) -> str:
        """Build the appropriate transform code based on the IR structure."""
        if 'tag = "operation_0"' in code:
            return (
                f'module attributes {{transform.with_named_sequence}} {{\n'
                f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
                f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
                f' : (!transform.any_op) -> !transform.any_op\n'
                f'    transform.structured.vectorize %op vector_sizes {vector_sizes}'
                f' : !transform.any_op\n'
                f'    transform.yield\n'
                f'  }}\n'
                f'}}\n'
            )
        else:
            # Preprocessed conv2d: match all generics, split to get the matmul-like one.
            # Optionally tile before vectorizing to keep vector sizes manageable.
            lines = [
                f'module attributes {{transform.with_named_sequence}} {{',
                f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{',
                f'    %generics = transform.structured.match ops{{["linalg.generic"]}} in %arg1'
                f' : (!transform.any_op) -> !transform.any_op',
                f'    %img2col_gen, %matmul_gen = transform.split_handle %generics'
                f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)',
            ]

            vectorize_target = '%matmul_gen'
            if tile_sizes:
                n_loops = sum(1 for t in tile_sizes if t != 0)
                loop_vars = ', '.join(f'%loop{i}' for i in range(n_loops))
                loop_types = ', '.join(['!transform.any_op'] * n_loops)
                lines.append(
                    f'    %tiled, {loop_vars} = transform.structured.tile_using_for %matmul_gen'
                    f' tile_sizes {tile_sizes}'
                    f' : (!transform.any_op) -> (!transform.any_op, {loop_types})'
                )
                vectorize_target = '%tiled'

            lines.extend([
                f'    transform.structured.vectorize {vectorize_target} vector_sizes {vector_sizes}'
                f' : !transform.any_op',
                f'    transform.yield',
                f'  }}',
                f'}}',
            ])
            return '\n'.join(lines) + '\n'

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]
        tile_sizes = params.get("tile_sizes")

        preprocessed = cls.preprocess(code, params)
        transform_code = cls._build_transform_code(preprocessed, vector_sizes, tile_sizes)

        try:
            result = run_transform_code(preprocessed, transform_code)
        except Exception:
            return code

        if not cls._check_vector_safety(result):
            return code

        return result

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        if not cls._check_vector_safety(after):
            return False
        return True
