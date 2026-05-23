import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Packing(ActionBase):
    """
    Reorder and pack operand data into a canonical contiguous layout that aligns
    with the vector access patterns required by downstream vectorization.
    transform.structured.pack is applied, then lower_pack lowers linalg.pack ops
    to pad + expand_shape + transpose so the result is immediately executable.
    Packing at different inner_tile sizes is a meaningful repeated application,
    so unique_execution = False.
    """

    unique_execution: bool = True

    VOCAB = [0, 16, 32, 64, 128]  # 0 = do not pack this loop dimension

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "description": (
                    "Per-loop-dimension pack sizes matching the linalg op's iteration space. "
                    "0 = do not pack that dimension."
                ),
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        packed_sizes = params.get("packed_sizes", [])
        if not packed_sizes:
            return False
        if all(s == 0 for s in packed_sizes):
            return False
        if any(s < 0 for s in packed_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        packed_sizes = params["packed_sizes"]
        if all(s == 0 for s in packed_sizes):
            return code

        sizes_str = str(packed_sizes)

        transform_code = (
            f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %packed = transform.structured.pack %op packed_sizes = {sizes_str}'
            f' : (!transform.any_op) -> (!transform.any_op)\n'
            f'    %pack_ops = transform.structured.match ops{{["linalg.pack"]}} in %arg1'
            f' : (!transform.any_op) -> !transform.op<"linalg.pack">\n'
            f'    %lowered_pad, %lowered_expand, %lowered_transpose ='
            f' transform.structured.lower_pack %pack_ops'
            f' : (!transform.op<"linalg.pack">)'
            f' -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %packed "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        # Packing should produce transpose or expand_shape ops
        if "linalg.transpose" not in after and "tensor.expand_shape" not in after:
            return False
        return True

    @classmethod
    def params_size(cls) -> int:
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"packed_sizes": sizes}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n = min(n_loops, MAX_PARAM_SLOTS)
        masks = []
        for i in range(n):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            slot_mask = np.array([
                s == 0 or (bound > 0 and bound % s == 0)
                for s in cls.VOCAB
            ], dtype=bool)
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        return np.concatenate(masks)
