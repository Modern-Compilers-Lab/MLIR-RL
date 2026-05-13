from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Peeling(ActionBase):
    """
    Separate partial or remainder iterations from the main loop body to allow the
    main loop to operate on full, uniform tiles. Tiles the operation first with a
    non-divisible tile size, then peels the outermost loop.

    This action tiles the target operation and then peels the outermost loop,
    separating the remainder iterations into a separate epilogue loop.
    """

    TILE_VOCAB = [24, 48, 36, 20, 40]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_size": {
                "description": "Tile size for outer dimension that may not evenly divide the iteration space.",
                "type": "int",
                "values": cls.TILE_VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_size = params.get("tile_size")
        if tile_size is None or tile_size <= 0:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_size = params["tile_size"]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:3 = transform.structured.tile_using_for %op tile_sizes [{tile_size}, {tile_size}, {tile_size}] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    %main_loop, %remainder = transform.loop.peel %loops#0 {{fail_if_already_divisible = false}} : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)\n'
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
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.TILE_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        tile_size = cls.TILE_VOCAB[raw_slots[0] % len(cls.TILE_VOCAB)]
        return {"tile_size": tile_size}
