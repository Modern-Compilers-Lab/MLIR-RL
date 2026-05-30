import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Packing(ActionBase):
    """Reorganize operand data into a blocked tile-contiguous layout.
    Uses transform.structured.pack to create hierarchical blocked format,
    then lowers pack/unpack ops to bufferizable tensor operations.
    """

    unique_execution = True  # restructures data layout; re-application is ill-defined

    VOCAB = [0, 4, 8, 16, 32, 64]  # per-dimension pack sizes; 0 = do not pack this dimension

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "type": "list[int]",
                "description": "Per-dimension inner tile sizes for packing (0 means skip)",
                "values": cls.VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        packed_sizes = params.get("packed_sizes", [])
        if not packed_sizes or all(s == 0 for s in packed_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        packed_sizes = params["packed_sizes"]

        # Pack, then lower all linalg.pack/unpack ops so the result is bufferizable.
        # transform.structured.pack returns a single handle (the packed linalg op).
        # We use transform.foreach to lower each pack/unpack individually.
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %packed_op = transform.structured.pack %op packed_sizes = {packed_sizes}'
            f' : (!transform.any_op) -> !transform.any_op\n'
            # Lower all linalg.pack ops to tensor.pad + tensor.expand_shape + linalg.transpose
            f'    %all_packs = transform.structured.match ops{{["linalg.pack"]}} in %arg1'
            f' : (!transform.any_op) -> !transform.op<"linalg.pack">\n'
            f'    transform.foreach %all_packs : !transform.op<"linalg.pack"> {{\n'
            f'    ^bb0(%pack: !transform.op<"linalg.pack">):\n'
            f'      %pad, %expand, %transpose = transform.structured.lower_pack %pack'
            f' : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)\n'
            f'      transform.yield\n'
            f'    }}\n'
            # Lower all linalg.unpack ops to tensor.empty + linalg.transpose + tensor.collapse_shape + tensor.extract_slice
            f'    %all_unpacks = transform.structured.match ops{{["linalg.unpack"]}} in %arg1'
            f' : (!transform.any_op) -> !transform.op<"linalg.unpack">\n'
            f'    transform.foreach %all_unpacks : !transform.op<"linalg.unpack"> {{\n'
            f'    ^bb0(%unpack: !transform.op<"linalg.unpack">):\n'
            f'      %empty, %ltranspose, %collapse, %extract = transform.structured.lower_unpack %unpack'
            f' : (!transform.op<"linalg.unpack">) -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">)\n'
            f'      transform.yield\n'
            f'    }}\n'
            # Tag the packed operation
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %packed_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
            m = np.array(
                [s == 0 or (bound > 0 and bound % s == 0) for s in cls.VOCAB],
                dtype=bool,
            )
            if not m.any():
                m[0] = True
            masks.append(m)
        return np.concatenate(masks)
