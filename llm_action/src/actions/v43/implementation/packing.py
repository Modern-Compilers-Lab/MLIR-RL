import numpy as np
from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Packing(ActionBase):
    """Reorganize operand data layout into blocked/panel format for contiguous tile access.

    Uses transform.structured.pack to create blocked layout, then lowers the pack/unpack
    ops to tensor operations that are bufferizable.

    Repeatable: different packing sizes can be applied at different levels.
    """

    unique_execution = False  # different packed sizes at different levels are meaningful

    VOCAB = [0, 4, 8, 16, 32, 64]  # 0 = do not pack that dimension

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "description": "Block size per iteration dimension for packed layout; 0 means skip",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        packed_sizes = params.get("packed_sizes", [])
        if not packed_sizes or all(s == 0 for s in packed_sizes):
            return False
        # Packing requires linalg ops
        if "linalg." not in code:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        packed_sizes = params["packed_sizes"]

        # Pack the target op, then lower pack/unpack ops to bufferizable form
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %packed = transform.structured.pack %op packed_sizes = {str(packed_sizes)}"
            f" : (!transform.any_op) -> !transform.any_op\n"
            # Lower pack ops
            f'    %all_packs = transform.structured.match ops{{["linalg.pack"]}} in %arg1'
            f" : (!transform.any_op) -> !transform.op<\"linalg.pack\">\n"
            f"    transform.foreach %all_packs : !transform.op<\"linalg.pack\"> {{\n"
            f"    ^bb0(%pack: !transform.op<\"linalg.pack\">):\n"
            f"      %pad, %expand, %transpose = transform.structured.lower_pack %pack"
            f" : (!transform.op<\"linalg.pack\">) -> (!transform.op<\"tensor.pad\">, !transform.op<\"tensor.expand_shape\">, !transform.op<\"linalg.transpose\">)\n"
            f"      transform.yield\n"
            f"    }}\n"
            # Lower unpack ops
            f'    %all_unpacks = transform.structured.match ops{{["linalg.unpack"]}} in %arg1'
            f" : (!transform.any_op) -> !transform.op<\"linalg.unpack\">\n"
            f"    transform.foreach %all_unpacks : !transform.op<\"linalg.unpack\"> {{\n"
            f"    ^bb0(%unpack: !transform.op<\"linalg.unpack\">):\n"
            f"      %empty, %tr, %collapse, %extract = transform.structured.lower_unpack %unpack"
            f" : (!transform.op<\"linalg.unpack\">) -> (!transform.op<\"tensor.empty\">, !transform.op<\"linalg.transpose\">, !transform.op<\"tensor.collapse_shape\">, !transform.op<\"tensor.extract_slice\">)\n"
            f"      transform.yield\n"
            f"    }}\n"
            # Re-tag the packed op
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %packed "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        n = min(n_loops, MAX_PARAM_SLOTS)
        return [len(cls.VOCAB)] * n

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
            slot_mask = np.array(
                [s == 0 or (bound > 0 and bound % s == 0) for s in cls.VOCAB],
                dtype=bool,
            )
            if not slot_mask.any():
                slot_mask[0] = True
            masks.append(slot_mask)
        return np.concatenate(masks)
