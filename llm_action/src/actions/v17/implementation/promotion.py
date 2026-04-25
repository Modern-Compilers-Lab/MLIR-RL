from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """
    Promotes operands of the tagged linalg operation by tiling and then
    padding the tiled operands into contiguous temporary buffers.
    Uses transform.structured.tile_using_for + transform.structured.pad.
    """

    PROMOTE_OPTIONS = [[0], [1], [2], [0, 1], [0, 1, 2]]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "operands_to_promote": {
                "description": "Indices of operands to promote (pad with nofold).",
                "type": "list[int]",
                "values": cls.PROMOTE_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        ops = params.get("operands_to_promote", [])
        if not ops or not isinstance(ops, list):
            return False
        if any(not isinstance(o, int) or o < 0 for o in ops):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        ops_to_promote = params["operands_to_promote"]

        # Build nofold_flags: 1 for promoted operands, 0 otherwise
        max_operand = max(ops_to_promote) + 1
        n_operands = max(max_operand, 3)
        nofold = [1 if i in ops_to_promote else 0 for i in range(n_operands)]
        nofold_str = str(nofold)

        padding_values_str = ", ".join(["0.0 : f64"] * n_operands)

        # Step 1: Tile to create subviews, then pad
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f" : (!transform.any_op) -> !transform.any_op\n"
            f"    %tiled_op, %loops:2 = transform.structured.tile_using_for %op"
            f" tile_sizes [32, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n"
            f"    %padded, %pad, %copy = transform.structured.pad %tiled_op {{\n"
            f"        padding_values = [{padding_values_str}],\n"
            f"        padding_dimensions = [0, 1, 2, 3],\n"
            f"        nofold_flags = {nofold_str},\n"
            f'        copy_back_op = "bufferization.materialize_in_destination"\n'
            f"    }} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %padded "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return [len(cls.PROMOTE_OPTIONS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        idx = raw_slots[0] % len(cls.PROMOTE_OPTIONS)
        return {"operands_to_promote": cls.PROMOTE_OPTIONS[idx]}
