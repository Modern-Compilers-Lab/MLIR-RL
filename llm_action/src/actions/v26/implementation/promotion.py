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


class Promotion(ActionBase):
    """Copy tiled operand data into contiguous temporary buffers.
    Internally tiles, bufferizes, promotes, and canonicalizes."""

    unique_execution = False  # Can promote different operand subsets

    OPERAND_OPTIONS = [[0, 1, 2], [0, 1], [0], [1], [2]]

    @classmethod
    def parameters(cls) -> dict:
        return {"operands_to_promote": "list of operand indices to copy into contiguous buffers"}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        operands = params.get("operands_to_promote", [])
        if not operands:
            return False
        # Only meaningful for ops with multiple operands
        has_multi_operands = (
            "linalg.matmul" in code
            or "linalg.conv_2d_nchw_fchw" in code
            or "linalg.pooling_nchw_max" in code
            or "linalg.generic" in code
        )
        if not has_multi_operands:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        operands = params["operands_to_promote"]
        n_loops = _count_loops(code)
        if n_loops == 0:
            return code

        # Choose tile sizes for internal tiling: tile first 2 dims by 32
        tile_sizes = [32, 32] + [0] * max(0, n_loops - 2)
        n_tiled = sum(1 for s in tile_sizes if s != 0)
        r = ', '.join(['!transform.any_op'] * n_tiled)
        operands_str = ', '.join(str(o) for o in operands)

        # Promotion sequence: tile -> tag -> bufferize (with func boundaries) ->
        # re-match -> promote -> canonicalize + CSE -> re-tag.
        # Uses transform.bufferization.one_shot_bufferize with
        # bufferize_function_boundaries=true to produce fully memref code
        # (no to_tensor/to_buffer at boundaries).
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %module : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_tiled} = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {r})\n'
            f'    %tiled_tag = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tiled_tag : !transform.any_op, !transform.any_param\n'
            f'    %m1 = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}} %module {{bufferize_function_boundaries = true}} : (!transform.any_op) -> !transform.any_op\n'
            f'    %target = transform.structured.match attributes{{tag = "tiled_target"}} in %m1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted = transform.structured.promote %target {{operands_to_promote = [{operands_str}]}} : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %promoted "tag" = %final_tag : !transform.any_op, !transform.any_param\n'
            f'    %m2 = transform.apply_registered_pass "canonicalize" to %m1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %m3 = transform.apply_registered_pass "cse" to %m2 : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            result = run_transform_code(code, transform_code)
        except Exception:
            return code

        # Strip memref.dealloc ops so that buffer-deallocation-pipeline
        # in the execution pipeline can insert correct deallocs.
        lines = result.split('\n')
        lines = [l for l in lines if 'memref.dealloc' not in l]
        return '\n'.join(lines)

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if after.strip() == before.strip():
            return False
        if "func.func" not in after:
            return False
        # Promoted code should have linalg.copy (data promotion pattern)
        if "linalg.copy" not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.OPERAND_OPTIONS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        idx = raw_slots[0] % len(cls.OPERAND_OPTIONS)
        return {"operands_to_promote": cls.OPERAND_OPTIONS[idx]}
