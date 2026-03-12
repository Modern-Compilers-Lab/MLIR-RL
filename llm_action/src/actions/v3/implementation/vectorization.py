import re
from functools import reduce
from operator import mul

from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """
    Map loop iterations onto SIMD vector lanes, producing vector operations that
    process multiple elements per instruction. Uses transform.structured.vectorize.
    Vector sizes must be >= corresponding iteration space sizes.
    """

    MAX_VECTOR_ELEMENTS = 1024
    MAX_VECTOR_RANK = 3

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "List of vector sizes, one per loop dimension. Each must be >= the corresponding iteration space size. Empty list means infer sizes automatically.",
                "type": "list[int]",
                "default": [],
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes", [])
        if vector_sizes:
            if not isinstance(vector_sizes, list):
                return False
            if not all(isinstance(s, int) and s > 0 for s in vector_sizes):
                return False
            total = reduce(mul, vector_sizes, 1)
            if total > cls.MAX_VECTOR_ELEMENTS:
                print(total)
                return False
            # if len(vector_sizes) > cls.MAX_VECTOR_RANK:
            #     for s in vector_sizes:
            #         if s > 16:
            #             return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def _check_vector_safety(cls, transformed_code: str) -> bool:
        vector_pattern = re.compile(r'vector<([^>]+)>')
        for match in vector_pattern.finditer(transformed_code):
            shape_str = match.group(1)
            dims_part = shape_str.split('x')
            numeric_dims = []
            for d in dims_part:
                d = d.strip()
                if d and d[0].isdigit():
                    numeric_dims.append(int(d))
            if len(numeric_dims) > 0:
                total = reduce(mul, numeric_dims, 1)
                if total > cls.MAX_VECTOR_ELEMENTS:
                    return False
                # if len(numeric_dims) > 3:
                #     return False
        return True

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params.get("vector_sizes", [])

        if vector_sizes:
            sizes_str = ", ".join(str(s) for s in vector_sizes)
            vectorize_line = f'    transform.structured.vectorize %op vector_sizes [{sizes_str}] : !transform.any_op'
        else:
            vectorize_line = '    transform.structured.vectorize %op : !transform.any_op'

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'{vectorize_line}\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
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
