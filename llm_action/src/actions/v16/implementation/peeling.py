from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Peeling(ActionBase):
    """
    Peeling action: separates boundary/remainder iterations from the main loop
    body by tiling a dimension and then peeling the resulting loop using
    transform.loop.peel. This allows the main body to be optimized with
    full-size tiles while the peeled remainder handles edge cases.
    """

    TILE_VOCAB = [4, 8, 16, 32]
    PEEL_FRONT_VOCAB = [0, 1]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_size": {
                "description": "Tile size for the dimension to peel",
                "type": "int",
                "values": cls.TILE_VOCAB,
            },
            "peel_front": {
                "description": "Whether to peel the first iteration (1) or last (0)",
                "type": "int",
                "values": cls.PEEL_FRONT_VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_size = params.get("tile_size", 0)
        if tile_size <= 0:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_size = params["tile_size"]
        peel_front = bool(params.get("peel_front", False))

        n_iterators = cls._count_iterators(code)
        # Tile the first dimension to create a loop
        tile_sizes = [tile_size] + [0] * (n_iterators - 1)

        peel_front_str = "true" if peel_front else "false"

        transform_code = (
            f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop = transform.structured.tile_using_for %op'
            f' tile_sizes {str(tile_sizes)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    %main_loop, %remainder_loop = transform.loop.peel %loop'
            f' {{peel_front = {peel_front_str}}}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
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
        return 2

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.TILE_VOCAB), len(cls.PEEL_FRONT_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        tile_size = cls.TILE_VOCAB[raw_slots[0] % len(cls.TILE_VOCAB)]
        peel_front = cls.PEEL_FRONT_VOCAB[raw_slots[1] % len(cls.PEEL_FRONT_VOCAB)]
        return {"tile_size": tile_size, "peel_front": peel_front}

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
