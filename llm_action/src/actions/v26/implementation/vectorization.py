import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


def _count_loops(code: str) -> int:
    if 'tag = "operation_0"' not in code:
        return 0
    if "linalg.matmul" in code:
        return 3
    if "linalg.conv_2d_nchw_fchw" in code:
        return 7
    if "linalg.pooling_nchw_max" in code:
        return 6
    m = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
    if m:
        return len([s.strip() for s in m.group(1).split(',')])
    m = re.search(r'outs\([^:]+:\s*tensor<([^>]+)>', code)
    if m:
        return len(m.group(1).split('x')) - 1
    return 0


class Vectorization(ActionBase):
    """Vectorize a linalg operation by tiling to vector-sized blocks then vectorizing.
    Tiles all dimensions to specified sizes, then applies vectorization."""

    unique_execution = True  # Lowers the linalg op to vector ops; second application has no valid target

    VOCAB = [2, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {"vector_sizes": "list of vector widths per dimension (applied to last N dims)"}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes", [])
        if not vector_sizes or all(s <= 0 for s in vector_sizes):
            return False
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
        vector_sizes = list(params["vector_sizes"])
        n_loops = _count_loops(code)
        if n_loops == 0:
            return code

        # Build full tile sizes: left-pad with 1 to match n_loops
        if len(vector_sizes) < n_loops:
            full_sizes = [1] * (n_loops - len(vector_sizes)) + vector_sizes
        else:
            full_sizes = vector_sizes[:n_loops]

        # Replace any 0 with 1
        full_sizes = [max(s, 1) for s in full_sizes]

        product = 1
        for s in full_sizes:
            product *= s
        if product > 1024:
            return code

        n_tiled = sum(1 for s in full_sizes if s > 1)
        if n_tiled == 0:
            # All sizes are 1, trivial vectorization
            n_tiled = n_loops
            # Still proceed: tile all dims by 1, vectorize

        # Use full_sizes as tile_sizes (all > 0 so all dims tiled)
        r = ', '.join(['!transform.any_op'] * n_loops)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {full_sizes} : (!transform.any_op) -> (!transform.any_op, {r})\n'
            f'    transform.structured.vectorize %tiled_op : !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
            # Validate vector sizes in result
            for m in re.finditer(r'vector<([^>]+)>', result):
                shape_str = m.group(1)
                dims = [int(d) for d in shape_str.split('x') if d.strip().isdigit()]
                if dims:
                    vec_product = 1
                    for d in dims:
                        vec_product *= d
                    if vec_product > 1024:
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
        # Enforce product constraint
        product = 1
        for s in sizes:
            product *= s
        while product > 1024 and sizes:
            max_idx = sizes.index(max(sizes))
            sizes[max_idx] = max(2, sizes[max_idx] // 2)
            product = 1
            for s in sizes:
                product *= s
        return {"vector_sizes": sizes}
