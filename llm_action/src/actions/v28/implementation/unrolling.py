from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """Tile to isolate a loop band, then unroll the innermost tiled loop.

    Repeatable: can unroll different loops at different nesting levels.
    """

    unique_execution: bool = False  # can unroll different loops

    TILE_VOCAB = [0, 4, 8, 16, 32]
    UNROLL_VOCAB = [2, 3, 4]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes to isolate the loop band before unrolling; 0 = do not tile.",
                "type": "list[int]",
                "values": cls.TILE_VOCAB,
            },
            "unroll_factor": {
                "description": "Factor by which to unroll the innermost tiled loop.",
                "type": "int",
                "values": cls.UNROLL_VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or not isinstance(tile_sizes, list):
            return False
        if all(s == 0 for s in tile_sizes):
            return False
        unroll_factor = params.get("unroll_factor", 0)
        if unroll_factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        unroll_factor = params["unroll_factor"]
        n_loops = sum(1 for s in tile_sizes if s != 0)
        assert n_loops > 0

        loop_results = ", ".join(["!transform.op<\"scf.for\">"] * n_loops)
        tile_sizes_str = str(tile_sizes)

        # Annotate the tiled op BEFORE unrolling (unroll invalidates nested handles)
        # Unroll the innermost (last) generated loop
        innermost_idx = n_loops - 1

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {tile_sizes_str}'
            f' : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.loop.unroll %loops#{innermost_idx} {{factor = {unroll_factor}}}'
            f' : !transform.op<"scf.for">\n'
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
        # tile_sizes use up to MAX_PARAM_SLOTS-1 slots, unroll_factor uses 1 slot
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        n_tile_slots = min(n_loops, MAX_PARAM_SLOTS - 1)
        tile_classes = [len(cls.TILE_VOCAB)] * n_tile_slots
        unroll_classes = [len(cls.UNROLL_VOCAB)]
        return tile_classes + unroll_classes

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n_tile_slots = min(n_loops, MAX_PARAM_SLOTS - 1)
        sizes = [cls.TILE_VOCAB[raw_slots[i] % len(cls.TILE_VOCAB)] for i in range(n_tile_slots)]
        unroll_idx = raw_slots[n_tile_slots] % len(cls.UNROLL_VOCAB) if n_tile_slots < len(raw_slots) else 0
        return {
            "tile_sizes": sizes,
            "unroll_factor": cls.UNROLL_VOCAB[unroll_idx],
        }
