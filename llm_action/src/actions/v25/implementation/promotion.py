from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """
    Copy tiled operand sub-tensors into contiguous temporary buffers to ensure
    stride-1 access within tiles. Requires internal bufferization as preprocessing.
    Canonicalization after promotion folds dynamic buffer shapes into static types.
    """

    unique_execution: bool = False  # can promote different operands at different tiling levels

    OPERAND_CONFIGS = [
        [0],
        [1],
        [0, 1],
        [0, 1, 2],
        [2],
    ]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "operands_to_promote": {
                "description": "List of operand indices to copy into contiguous local buffers.",
                "type": "list[int]",
                "values": cls.OPERAND_CONFIGS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        ops = params.get("operands_to_promote", [])
        if not ops or not isinstance(ops, list):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        operands = params["operands_to_promote"]
        num_operands = 3  # matmul/conv/pooling/add all have 3 operands (2 ins + 1 out)

        # Build pack_paddings: 1 for promoted operands, 0 otherwise
        pack_paddings = [1 if i in operands else 0 for i in range(num_operands)]
        pack_paddings_str = "[" + ", ".join(str(p) for p in pack_paddings) + "]"
        padding_values_str = "[" + ", ".join(["0.0 : f64"] * num_operands) + "]"

        # Use pad (tensor-level promotion): tile, then pad selected operands
        # into contiguous copies to ensure stride-1 access within tiles
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op0 = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loop0 = transform.structured.tile_using_for %op0 tile_sizes [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %padded, %pad, %copy = transform.structured.pad %tiled_op {{padding_values = {padding_values_str}, padding_dimensions = [0], pack_paddings = {pack_paddings_str}}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %padded "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return [len(cls.OPERAND_CONFIGS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        idx = raw_slots[0] % len(cls.OPERAND_CONFIGS)
        return {"operands_to_promote": cls.OPERAND_CONFIGS[idx]}
