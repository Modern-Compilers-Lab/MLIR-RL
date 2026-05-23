import re
import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Map innermost loop iterations of the contraction to SIMD vector operations.
    Lowering transform: consumes the linalg op and produces vector ops inside scf.for loops."""

    unique_execution: bool = True  # lowering: the linalg op is consumed, no valid target for a second application

    VOCAB = [1, 2, 4, 8, 16]  # vector sizes per dimension; all must be > 0

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Per-loop tile/vector sizes. All must be > 0. Product must be <= 2048.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or not isinstance(tile_sizes, list):
            return False
        if any(not isinstance(s, int) or s <= 0 for s in tile_sizes):
            return False
        product = 1
        for s in tile_sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        if product == 1:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        # All tile sizes for vectorization are > 0, so all create loops
        n_loops = len(tile_sizes)
        loop_results = ", ".join(["!transform.any_op"] * n_loops)

        tile_result_type = f"(!transform.any_op, {loop_results})"
        tile_line = (
            f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op "
            f"tile_sizes {tile_sizes} : (!transform.any_op) -> {tile_result_type}\n"
        )
        vectorize_line = (
            f"    transform.structured.vectorize %tiled_op vector_sizes {tile_sizes} : !transform.any_op\n"
        )
        # Tag the outermost loop (Category B lowering)
        tag_line = (
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f"    transform.annotate %loops#0 \"tag\" = %tag : !transform.any_op, !transform.any_param\n"
        )

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"{tile_line}"
            f"{vectorize_line}"
            f"{tag_line}"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )

        try:
            result = run_transform_code(code, transform_code)
        except Exception:
            return code

        # Post-transform safety: check vector sizes don't exceed limits
        vector_pattern = re.compile(r"vector<([^>]+)>")
        for match in vector_pattern.finditer(result):
            dims_str = match.group(1)
            # Extract numeric dimensions (skip type like f64, f32)
            parts = dims_str.replace("x", " ").split()
            dims = []
            for p in parts:
                try:
                    dims.append(int(p))
                except ValueError:
                    pass
            if dims:
                product = 1
                for d in dims:
                    product *= d
                if product > VECTORIZATION_SIZE_LIMIT:
                    return code
                if len(dims) >= 3 and product > 64:
                    return code

        return result

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
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        # Clamp product to VECTORIZATION_SIZE_LIMIT
        product = 1
        for s in sizes:
            product *= s
        while product > VECTORIZATION_SIZE_LIMIT and any(s > 1 for s in sizes):
            for j in range(len(sizes) - 1, -1, -1):
                if sizes[j] > 1:
                    idx = cls.VOCAB.index(sizes[j])
                    sizes[j] = cls.VOCAB[max(0, idx - 1)]
                    break
            product = 1
            for s in sizes:
                product *= s
        return {"tile_sizes": sizes}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n = min(n_loops, MAX_PARAM_SLOTS)
        masks = []
        for i in range(n):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            slot_mask = np.array(
                [bound > 0 and bound % s == 0 for s in cls.VOCAB],
                dtype=bool,
            )
            if not slot_mask.any():
                slot_mask[0] = True  # size=1 always valid
            masks.append(slot_mask)
        return np.concatenate(masks)
