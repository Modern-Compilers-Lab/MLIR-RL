from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Peeling(ActionBase):
    """Split a loop into a main body and remainder using transform.loop.peel."""

    PREPROCESS_TILE_SIZES = [24, 24, 24]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "peel_front": {
                "description": "If True, peel the first iteration; otherwise peel the last.",
                "type": "bool",
                "values": [False, True],
            }
        }

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
        code = cls.preprocess(code, params)
        peel_front = params.get("peel_front", False)
        peel_front_str = "true" if peel_front else "false"

        has_loops = "scf.for" in code

        if has_loops:
            # Get the parent scf.for of the tagged op and peel it
            transform_code = (
                f"module attributes {{transform.with_named_sequence}} {{\n"
                f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
                f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f'    %parent = transform.get_parent_op %op {{op_name = "scf.for"}} : (!transform.any_op) -> !transform.op<"scf.for">\n'
                f"    %peeled, %remainder = transform.loop.peel %parent {{peel_front = {peel_front_str}, fail_if_already_divisible = false}}"
                f' : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)\n'
                f"    transform.yield\n"
                f"  }}\n"
                f"}}\n"
            )
        else:
            # Tile with non-divisible sizes to create a remainder, then peel
            tile_sizes = cls.PREPROCESS_TILE_SIZES
            n_loops = len(tile_sizes)
            loop_types = ", ".join(['!transform.op<"scf.for">'] * n_loops)
            sizes_str = str(tile_sizes)

            transform_code = (
                f"module attributes {{transform.with_named_sequence}} {{\n"
                f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
                f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {sizes_str}"
                f" : (!transform.any_op) -> (!transform.any_op, {loop_types})\n"
                f"    %peeled, %remainder = transform.loop.peel %loops#{n_loops - 1}"
                f" {{peel_front = {peel_front_str}, fail_if_already_divisible = false}}"
                f' : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)\n'
                f"    transform.yield\n"
                f"  }}\n"
                f"}}\n"
            )

        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if before.strip() == after.strip():
            return False
        if "func.func" not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [2]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {"peel_front": bool(raw_slots[0] % 2)}
