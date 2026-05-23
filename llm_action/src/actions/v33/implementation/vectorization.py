from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code
import numpy as np
import re


class Vectorization(ActionBase):
    """Tile to vector-friendly sizes and vectorize the tiled linalg op.
    Lowering (Category B): the linalg op is consumed and replaced by vector ops.
    Note: conv2d with windowed access patterns cannot be directly vectorized;
    apply Im2colLowering first to convert to a matmul-like contraction."""

    # Vectorization consumes the linalg op; a second application has no valid target.
    unique_execution: bool = True

    VOCAB = [1, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Per-loop-dimension vector/tile sizes for vectorization; all must be > 0.",
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
        # Check vector product within safety limit
        product = 1
        for s in tile_sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]

        # Count non-one tile sizes to determine how many loops are generated
        n_loops = sum(1 for s in tile_sizes if s != 1)

        if n_loops == 0:
            # All sizes are 1 — tile to trivial sizes and vectorize with [1,...,1]
            # Still generates a tiled op but no loops
            result_types = "!transform.any_op"
            tile_result = "%tiled_op = transform.structured.tile_using_for %op tile_sizes " + str(tile_sizes)
            tile_result += f" : (!transform.any_op) -> ({result_types})\n"
            tag_target = "%tiled_op"
        else:
            result_types = ", ".join(["!transform.any_op"] * (1 + n_loops))
            tile_result = f"%tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {tile_sizes}"
            tile_result += f" : (!transform.any_op) -> ({result_types})\n"
            tag_target = "%loops#0"

        transform_code = (
            "module attributes {transform.with_named_sequence} {\n"
            "  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n"
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    {tile_result}"
            f"    transform.structured.vectorize %tiled_op vector_sizes {tile_sizes} : !transform.any_op\n"
            '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate {tag_target} "tag" = %tag : !transform.any_op, !transform.any_param\n'
            "    transform.yield\n"
            "  }\n"
            "}\n"
        )

        try:
            result = run_transform_code(code, transform_code)
            # Post-transform vector safety check
            if not cls._check_vector_safety(result):
                return code
            return result
        except Exception:
            return code

    @classmethod
    def _check_vector_safety(cls, code: str) -> bool:
        """Check that generated vectors are within safety limits."""
        vector_pattern = re.compile(r"vector<([^>]+)>")
        for match in vector_pattern.finditer(code):
            dims_str = match.group(1)
            # Remove type suffix (e.g., "xf64", "xf32", "xi32")
            parts = dims_str.split("x")
            dim_parts = []
            for p in parts:
                try:
                    dim_parts.append(int(p))
                except ValueError:
                    break
            if not dim_parts:
                continue
            product = 1
            for d in dim_parts:
                product *= d
            if product > VECTORIZATION_SIZE_LIMIT:
                return False
            if len(dim_parts) >= 3 and product > 64:
                return False
        return True

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
        # Clamp product to VECTORIZATION_SIZE_LIMIT
        product = 1
        for s in sizes:
            product *= s
        while product > VECTORIZATION_SIZE_LIMIT and sizes:
            # Reduce largest dimension
            max_idx = max(range(len(sizes)), key=lambda i: sizes[i])
            if sizes[max_idx] <= 1:
                break
            idx = cls.VOCAB.index(sizes[max_idx])
            sizes[max_idx] = cls.VOCAB[max(0, idx - 1)]
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
                slot_mask[0] = True  # size 1 is always safe
            masks.append(slot_mask)
        return np.concatenate(masks)
