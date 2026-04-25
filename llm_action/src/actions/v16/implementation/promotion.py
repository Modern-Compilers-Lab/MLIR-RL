from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """
    Promotion action: tiles the operation and pads the tiled operands into
    local buffers for better cache behavior using transform.structured.pad.
    This is the tensor-semantics equivalent of buffer promotion.
    """

    VOCAB = [0, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for pre-tiling before padding (0 = do not tile)",
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

        n_iterators = cls._count_iterators(code)
        full_sizes = list(tile_sizes) + [0] * max(0, n_iterators - len(tile_sizes))
        full_sizes = full_sizes[:n_iterators]

        n_loops = sum(1 for s in full_sizes if s != 0)
        if n_loops == 0:
            return code

        loop_handles = ", ".join([f"%loop{i}" for i in range(n_loops)])
        r = ", ".join(["!transform.any_op"] * (1 + n_loops))

        # Detect element type for padding value
        pad_val = "0.0 : f64"
        if "xf32>" in code or "xf32," in code:
            pad_val = "0.0 : f32"

        # Build padding dimensions list (all dims of the tiled op)
        pad_dims = list(range(min(4, n_iterators)))
        pad_dims_str = ", ".join(str(d) for d in pad_dims)

        transform_code = (
            f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, {loop_handles} = transform.structured.tile_using_for %op'
            f' tile_sizes {str(full_sizes)} : (!transform.any_op) -> ({r})\n'
            f'    %padded, %pad, %copy = transform.structured.pad %tiled_op'
            f' {{padding_values = [{pad_val}, {pad_val}, {pad_val}],'
            f' padding_dimensions = [{pad_dims_str}]}}'
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
