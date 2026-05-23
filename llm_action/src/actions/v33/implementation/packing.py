from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code
import numpy as np


class Packing(ActionBase):
    """Pack selected iterator dimensions of a linalg op for contiguous data layout.
    Structure-preserving (Category A): the result is a packed linalg.generic.
    Includes lower_pack/lower_unpack so output is bufferizable."""

    # Repeated packing at different granularities is a valid tuning knob.
    unique_execution: bool = True

    VOCAB = [0, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "description": "Per-iterator-dimension pack sizes; 0 means do not pack that dimension.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        packed_sizes = params.get("packed_sizes", [])
        if not packed_sizes or not isinstance(packed_sizes, list):
            return False
        if all(s == 0 for s in packed_sizes):
            return False
        if any(not isinstance(s, int) or s < 0 for s in packed_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        packed_sizes = params["packed_sizes"]

        transform_code = (
            "module attributes {transform.with_named_sequence} {\n"
            "  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n"
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %packed_op = transform.structured.pack %op packed_sizes = {packed_sizes} : (!transform.any_op) -> (!transform.any_op)\n"
            '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            '    transform.annotate %packed_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
            '    %packs = transform.structured.match ops{["linalg.pack"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.pack">\n'
            '    %pad, %expand, %transpose = transform.structured.lower_pack %packs : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)\n'
            '    %unpacks = transform.structured.match ops{["linalg.unpack"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.unpack">\n'
            '    %empty, %t2, %collapse, %extract = transform.structured.lower_unpack %unpacks : (!transform.op<"linalg.unpack">) -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">)\n'
            "    transform.yield\n"
            "  }\n"
            "}\n"
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
