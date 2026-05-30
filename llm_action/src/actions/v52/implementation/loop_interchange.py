import re
from itertools import permutations

import numpy as np

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


def _get_n_loops(code: str) -> int:
    """Extract number of loop dimensions from iterator_types in the code."""
    match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
    if not match:
        return 0
    return len([t.strip() for t in match.group(1).split(',')])


class LoopInterchange(ActionBase):
    """Reorder loop dimensions via generalize + interchange.

    Structure-preserving (Category A): output is still a linalg.generic op.
    Includes generalize step for robustness with named linalg ops.
    """

    unique_execution = True

    @classmethod
    def _get_candidates(cls, n_loops: int) -> list[list[int]]:
        """Generate non-identity permutations, capped at MAX_VOCAB_SIZE_PER_SLOT."""
        if n_loops < 2:
            return []
        identity = list(range(n_loops))
        candidates = [list(p) for p in permutations(range(n_loops)) if list(p) != identity]
        return candidates[:MAX_VOCAB_SIZE_PER_SLOT]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "Permutation of loop dimensions [0..n_loops-1].",
                "type": "list[int]",
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        perm = params.get("permutation", [])
        if not perm or len(perm) < 2:
            return False
        n = len(perm)
        if sorted(perm) != list(range(n)):
            return False
        if perm == list(range(n)):
            return False  # identity = no-op
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        perm = list(params["permutation"])
        n_loops = _get_n_loops(code)

        # Adjust permutation to match actual loop count
        if n_loops > 0 and len(perm) != n_loops:
            if len(perm) < n_loops:
                # Extend with identity for remaining dims
                perm = perm + list(range(len(perm), n_loops))
            else:
                perm = perm[:n_loops]
                # Validate truncated permutation is still valid
                if sorted(perm) != list(range(n_loops)):
                    return code

        perm_str = str(perm).replace(' ', '')

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %generic iterator_interchange = {perm_str} : (!transform.any_op) -> !transform.any_op\n'
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
        if 'func.func' not in after:
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
