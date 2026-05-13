from itertools import permutations

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """Permute the ordering of loops to improve memory access locality.

    Repeatable: different permutations at different nesting levels after
    multi-level tiling is a valid tuning strategy."""

    unique_execution: bool = True  # can apply different permutations after re-tiling

    @classmethod
    def _get_candidates(cls, n_loops: int) -> list[list[int]]:
        """Generate non-identity permutations, capped at MAX_VOCAB_SIZE_PER_SLOT."""
        identity = list(range(n_loops))
        candidates = []
        for perm in permutations(range(n_loops)):
            p = list(perm)
            if p != identity:
                candidates.append(p)
                if len(candidates) >= MAX_VOCAB_SIZE_PER_SLOT:
                    break
        if not candidates:
            candidates = [identity]
        return candidates

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "Reordering of loop indices (e.g., [1,0,2] swaps first two loops)",
                "type": "list[int]",
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        perm = params.get("permutation", [])
        if not perm:
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
            'module attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n'
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            '    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n'
            '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            '    transform.annotate %generic "tag" = %tag : !transform.any_op, !transform.any_param\n'
            '    transform.yield\n'
            '  }\n'
            '}\n'
        )
        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        preprocessed = cls.preprocess(code, params)
        perm = params["permutation"]
        perm_str = str(perm).replace(" ", "")

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %op iterator_interchange = {perm_str} : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %interchanged "tag" = %tag : !transform.any_op, !transform.any_param\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )

        try:
            return run_transform_code(preprocessed, transform_code)
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
        return [len(candidates)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        candidates = cls._get_candidates(n_loops)
        idx = raw_slots[0] % len(candidates)
        return {"permutation": candidates[idx]}
