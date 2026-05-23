from itertools import combinations

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code
import numpy as np


class LoopInterchange(ActionBase):
    """Permute loop order of a linalg op via generalize + interchange.
    Structure-preserving (Category A): the result is a linalg.generic."""

    # Repeated interchange at different nesting levels is a valid tuning knob.
    unique_execution: bool = True

    @classmethod
    def _get_candidates(cls, n_loops: int) -> list[list[int]]:
        """Generate single-swap permutations, capped at MAX_VOCAB_SIZE_PER_SLOT."""
        identity = list(range(n_loops))
        candidates = []
        for i, j in combinations(range(n_loops), 2):
            perm = identity.copy()
            perm[i], perm[j] = perm[j], perm[i]
            candidates.append(perm)
        return candidates[:MAX_VOCAB_SIZE_PER_SLOT]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "Permutation of loop indices defining the new iterator order.",
                "type": "list[int]",
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        perm = params.get("permutation", [])
        if not perm or not isinstance(perm, list):
            return False
        n = len(perm)
        if sorted(perm) != list(range(n)):
            return False
        if perm == list(range(n)):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """Generalize named linalg ops to linalg.generic (required for interchange)."""
        transform_code = (
            "module attributes {transform.with_named_sequence} {\n"
            "  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n"
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            "    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n"
            '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            '    transform.annotate %generic "tag" = %tag : !transform.any_op, !transform.any_param\n'
            "    transform.yield\n"
            "  }\n"
            "}\n"
        )
        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        perm = params["permutation"]
        # Generalize first (preprocessing)
        code = cls.preprocess(code, params)

        transform_code = (
            "module attributes {transform.with_named_sequence} {\n"
            "  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n"
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %interchanged = transform.structured.interchange %op iterator_interchange = {perm} : (!transform.any_op) -> !transform.any_op\n"
            '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            '    transform.annotate %interchanged "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        candidates = cls._get_candidates(n_loops)
        if not candidates:
            return [1]
        return [len(candidates)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        candidates = cls._get_candidates(n_loops)
        if not candidates:
            return {"permutation": list(range(n_loops))}
        idx = raw_slots[0] % len(candidates)
        return {"permutation": candidates[idx]}

    @classmethod
    def valid_param_mask(cls, n_loops: int, loop_bounds: list[int]) -> "np.ndarray | None":
        return None
