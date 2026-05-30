import numpy as np
from itertools import permutations

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """Reorder loop dimensions by applying a permutation to the iterator indices.

    Requires generalization to linalg.generic first (interchange only works on generic ops).
    """

    unique_execution = False  # Can be applied at different nesting levels after re-tiling

    @classmethod
    def _get_candidates(cls, n_loops: int) -> list[list[int]]:
        """Generate non-identity permutations, capped at MAX_VOCAB_SIZE_PER_SLOT."""
        identity = list(range(n_loops))
        candidates = [list(p) for p in permutations(range(n_loops)) if list(p) != identity]
        if len(candidates) > MAX_VOCAB_SIZE_PER_SLOT:
            # Prioritize: reverse, swap-first-two, and rotations
            priority = []
            # Reverse
            rev = list(reversed(identity))
            if rev != identity:
                priority.append(rev)
            # Swap first two
            if n_loops >= 2:
                swap = identity.copy()
                swap[0], swap[1] = swap[1], swap[0]
                if swap not in priority:
                    priority.append(swap)
            # Swap last two
            if n_loops >= 2:
                swap = identity.copy()
                swap[-1], swap[-2] = swap[-2], swap[-1]
                if swap not in priority:
                    priority.append(swap)
            # Rotations
            for k in range(1, n_loops):
                rot = identity[k:] + identity[:k]
                if rot != identity and rot not in priority:
                    priority.append(rot)
            # Fill remaining from candidates
            for c in candidates:
                if len(priority) >= MAX_VOCAB_SIZE_PER_SLOT:
                    break
                if c not in priority:
                    priority.append(c)
            candidates = priority[:MAX_VOCAB_SIZE_PER_SLOT]
        return candidates if candidates else [[0]]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "A non-identity permutation of loop dimension indices.",
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
        # Must be a valid permutation
        if sorted(perm) != list(range(len(perm))):
            return False
        # Must be non-identity
        if perm == list(range(len(perm))):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        perm = params["permutation"]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %gen = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %gen iterator_interchange = {perm} : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %interchanged "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        candidates = cls._get_candidates(n_loops)
        return [len(candidates)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        candidates = cls._get_candidates(n_loops)
        idx = raw_slots[0] % len(candidates)
        return {"permutation": candidates[idx]}
