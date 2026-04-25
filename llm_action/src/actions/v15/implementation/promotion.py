import re

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """
    Tile the target operation and then pad operand slices, materializing
    them into local static-sized tensors. This is the tensor-semantic
    equivalent of operand promotion to local buffers.
    """

    VOCAB = [0, 4, 8, 16, 32]

    @classmethod
    def _detect_element_type(cls, code: str) -> str:
        m = re.search(r'tensor<[^>]*x(f16|f32|f64|bf16)>', code)
        return m.group(1) if m else "f64"

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for the sub-tile whose operands are promoted. 0 means no tiling.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or all(s == 0 for s in tile_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_loops = sum(1 for s in tile_sizes if s != 0)
        if n_loops == 0:
            return code

        etype = cls._detect_element_type(code)
        loop_types = ", ".join(["!transform.any_op"] * n_loops)
        tile_sizes_str = str(tile_sizes).replace("'", "")

        n_operands = 3
        padding_values_str = ", ".join([f"0.0 : {etype}"] * n_operands)
        padding_dims_str = ", ".join(str(i) for i in range(len(tile_sizes)))

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op'
            f' tile_sizes {tile_sizes_str}'
            f' : (!transform.any_op) -> (!transform.any_op, {loop_types})\n'
            f'    %padded, %pad, %copy = transform.structured.pad %tiled_op'
            f' {{padding_values = [{padding_values_str}],'
            f' padding_dimensions = [{padding_dims_str}]}}'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)\n'
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
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"tile_sizes": sizes}
