from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopPeeling(ActionBase):
    """
    Separate boundary iterations from the main loop to enable clean tiling,
    vectorization, or parallelization of the remaining iterations.

    Includes tiling with non-divisible tile sizes as preprocessing to create
    loops with remainders, then peels the outermost loop.
    Repeated application to peel different loops is meaningful.
    """

    unique_execution: bool = False  # can peel different loops at different levels

    # Fixed tile sizes chosen to NOT evenly divide common matmul dimensions,
    # ensuring there are remainder iterations to peel.
    # 48 does not divide 128 (128 = 2*48 + 32) or 256 (256 = 5*48 + 16)
    TILE_SIZES = [48, 48, 48]

    @classmethod
    def parameters(cls) -> dict:
        return {}

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = cls.TILE_SIZES
        n_loops = len(tile_sizes)
        loop_handles = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_handles})\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    %cast_loop = transform.cast %loops#0 : !transform.any_op to !transform.op<"scf.for">\n'
            f'    %main_loop, %remainder_loop = transform.loop.peel %cast_loop {{peel_front = false}} : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)\n'
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
        return 0

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return []

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {}
