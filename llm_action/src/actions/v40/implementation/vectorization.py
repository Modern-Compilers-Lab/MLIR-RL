import re

import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT, VECTORIZATION_SIZE_LIMIT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Map innermost loop iterations to SIMD vector lanes (AVX2 FMA).
    Preprocessing tiles to vector sizes before SIMD lowering.
    Mode 0 = sequential preprocessing tiling, Mode 1 = parallel preprocessing tiling.
    """

    unique_execution = True  # consumes the linalg op, lowering to vector + scf ops

    VOCAB = [1, 2, 4, 8, 16]  # per-dim vector sizes (f64: 4 lanes per 256-bit register)
    MODE_VOCAB = [0, 1]  # 0 = sequential tiling, 1 = parallel tiling

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "type": "list[int]",
                "description": "Per-loop vector sizes for SIMD lowering",
                "values": cls.VOCAB,
            },
            "parallelize": {
                "type": "int",
                "description": "0 = sequential preprocessing, 1 = parallel preprocessing",
                "values": cls.MODE_VOCAB,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vector_sizes = params.get("vector_sizes", [])
        if not vector_sizes:
            return False
        if all(s == 1 for s in vector_sizes):
            return False
        product = 1
        for s in vector_sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vector_sizes = params["vector_sizes"]
        parallelize = params.get("parallelize", 0)

        n_loops = len(vector_sizes)

        if parallelize == 0:
            # Sequential: tile_using_for + vectorize, tag outermost loop
            loop_types = ", ".join(["!transform.any_op"] * n_loops)
            transform_code = (
                f"module attributes {{transform.with_named_sequence}} {{\n"
                f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
                f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f"    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {vector_sizes}"
                f" : (!transform.any_op) -> (!transform.any_op, {loop_types})\n"
                f"    transform.structured.vectorize %tiled_op vector_sizes {vector_sizes} : !transform.any_op\n"
                f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
                f'    transform.annotate %loops#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
                f"    transform.yield\n"
                f"  }}\n"
                f"}}\n"
            )
        else:
            # Parallel: tile_using_forall + vectorize, tag the forall
            transform_code = (
                f"module attributes {{transform.with_named_sequence}} {{\n"
                f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
                f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f"    %tiled_op, %forall = transform.structured.tile_using_forall %op tile_sizes {vector_sizes}"
                f" : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n"
                f"    transform.structured.vectorize %tiled_op vector_sizes {vector_sizes} : !transform.any_op\n"
                f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
                f'    transform.annotate %forall "tag" = %tag : !transform.any_op, !transform.any_param\n'
                f"    transform.yield\n"
                f"  }}\n"
                f"}}\n"
            )

        try:
            result = run_transform_code(code, transform_code)
            # Validate vector sizes in output per vectorization safety contract
            for match in re.finditer(r"vector<([^>]+)>", result):
                dims_str = match.group(1)
                dims_parts = dims_str.split("x")
                try:
                    numeric = [
                        int(d)
                        for d in dims_parts
                        if d.strip() not in ("f32", "f64", "f16", "bf16", "i8", "i16", "i32", "i64", "index")
                    ]
                    if numeric:
                        product = 1
                        for s in numeric:
                            product *= s
                        if product > VECTORIZATION_SIZE_LIMIT:
                            return code
                        if len(numeric) >= 3 and product > 64:
                            return code
                except ValueError:
                    continue
            return result
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
        n = min(n_loops, MAX_PARAM_SLOTS - 1)
        return [len(cls.MODE_VOCAB)] + [len(cls.VOCAB)] * n

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS - 1)
        parallelize = cls.MODE_VOCAB[raw_slots[0] % len(cls.MODE_VOCAB)]
        sizes = [cls.VOCAB[raw_slots[1 + i] % len(cls.VOCAB)] for i in range(n)]
        # Clamp total vector product to stay within safety limit
        product = 1
        for s in sizes:
            product *= s
        if product > VECTORIZATION_SIZE_LIMIT:
            sizes = [min(s, 4) for s in sizes]
        return {"vector_sizes": sizes, "parallelize": parallelize}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        if not loop_bounds:
            return None
        n = min(n_loops, MAX_PARAM_SLOTS - 1)
        masks = [np.ones(len(cls.MODE_VOCAB), dtype=bool)]  # mode slot: always valid
        for i in range(n):
            bound = loop_bounds[i] if i < len(loop_bounds) else 0
            m = np.array(
                [bound > 0 and bound % s == 0 for s in cls.VOCAB],
                dtype=bool,
            )
            if not m.any():
                m[0] = True  # fallback: allow size 1
            masks.append(m)
        return np.concatenate(masks)
