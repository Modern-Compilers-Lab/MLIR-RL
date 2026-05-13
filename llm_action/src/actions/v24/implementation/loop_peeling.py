from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopPeeling(ActionBase):
    """
    Tile the reduction (innermost) dimension of the target operation and then
    peel the last partial iteration of the resulting loop. This produces a
    main loop with clean divisible bounds and a remainder loop, enabling
    clean vectorization of the main body.

    The action tiles the K (reduction) dimension then peels the generated loop.

    unique_execution = True because peeling the same loop twice is a no-op;
    the structural change is one-shot.
    """

    unique_execution: bool = True  # peeling the same loop twice is meaningless

    VOCAB = [3, 5, 7, 11, 13]  # tile sizes that don't divide common dims evenly

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_size": {
                "description": "Tile size for the reduction dimension before peeling. "
                               "Should not evenly divide the dimension for peeling to be effective.",
                "type": "int",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_size = params.get("tile_size")
        if tile_size is None or not isinstance(tile_size, int) or tile_size < 2:
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
            f'    %tiled_op, %loop0 = transform.structured.tile_using_for %op tile_sizes [0, 0, {tile_size}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    %cast_loop = transform.cast %loop0 : !transform.any_op to !transform.op<"scf.for">\n'
            f'    %main_loop, %remainder = transform.loop.peel %cast_loop {{fail_if_already_divisible = false}} : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)\n'
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
        return [len(cls.VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        idx = raw_slots[0] % len(cls.VOCAB)
        return {"tile_size": cls.VOCAB[idx]}
