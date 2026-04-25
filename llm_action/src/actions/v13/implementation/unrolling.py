from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Unrolling(ActionBase):
    """Unroll the innermost loop of a tiled operation."""

    VOCAB = [2, 4, 8, 16, 32]
    PREPROCESS_TILE_SIZES = [32, 32, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "unroll_factor": {
                "description": "Number of times to replicate the loop body.",
                "type": "int",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
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
        code = cls.preprocess(code, params)
        unroll_factor = params["unroll_factor"]

        has_loops = "scf.for" in code

        if has_loops:
            # Code already has loops - find the tagged op's parent loop and unroll it
            transform_code = (
                f"module attributes {{transform.with_named_sequence}} {{\n"
                f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
                f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f'    %parent = transform.get_parent_op %op {{op_name = "scf.for"}} : (!transform.any_op) -> !transform.any_op\n'
                f"    transform.loop.unroll %parent {{factor = {unroll_factor}}} : !transform.any_op\n"
                f"    transform.yield\n"
                f"  }}\n"
                f"}}\n"
            )
        else:
            # No loops - tile first to create loops, then unroll the innermost
            tile_sizes = cls.PREPROCESS_TILE_SIZES
            n_loops = len(tile_sizes)
            loop_types = ", ".join(["!transform.any_op"] * n_loops)
            sizes_str = str(tile_sizes)

            transform_code = (
                f"module attributes {{transform.with_named_sequence}} {{\n"
                f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
                f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {sizes_str}"
                f" : (!transform.any_op) -> (!transform.any_op, {loop_types})\n"
                f"    transform.loop.unroll %loops#{n_loops - 1} {{factor = {unroll_factor}}} : !transform.any_op\n"
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
        return [len(cls.VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {"unroll_factor": cls.VOCAB[raw_slots[0] % len(cls.VOCAB)]}
