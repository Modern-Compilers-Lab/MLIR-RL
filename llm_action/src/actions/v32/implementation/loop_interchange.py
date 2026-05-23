import numpy as np
from itertools import permutations

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """
    Permute the order of loops in a loop nest to improve spatial locality.
    Different permutations at different nesting levels are meaningful, so
    unique_execution = False.
    """

    unique_execution: bool = True

    @classmethod
    def _get_candidates(cls, n_loops: int) -> list[list[int]]:
        """Return all non-identity permutations for n_loops, capped at MAX_VOCAB_SIZE_PER_SLOT."""
        if n_loops < 2:
            return []
        identity = tuple(range(n_loops))
        all_perms = list(permutations(range(n_loops)))
        non_identity = [list(p) for p in all_perms if p != identity]
        return non_identity[:MAX_VOCAB_SIZE_PER_SLOT]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "Loop dimension permutation (index array). Must be a valid permutation of [0..n_loops-1].",
                "type": "list[int]",
                "values": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        permutation = params.get("permutation", [])
        if not permutation:
            return False
        n = len(permutation)
        if n < 2:
            return False
        # Validate it's a proper permutation
        if sorted(permutation) != list(range(n)):
            return False
        # Identity permutation is a no-op
        if permutation == list(range(n)):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        permutation = params["permutation"]
        perm_str = str(permutation)

        transform_code = (
            f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %generic'
            f' iterator_interchange = {perm_str} : (!transform.any_op) -> !transform.any_op\n'
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
        # All permutations are syntactically valid; no divisibility constraint.
        return None
