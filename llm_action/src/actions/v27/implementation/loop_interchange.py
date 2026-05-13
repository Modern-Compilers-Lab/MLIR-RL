from itertools import permutations

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """Permute loop iterators via generalize + interchange.

    Reorders the loops within a linalg.generic operation to improve memory
    access stride patterns and enable more efficient vectorization.
    Requires generalization as preprocessing (linalg.matmul -> linalg.generic).
    """

    # Different permutations at different nesting levels is a meaningful tuning knob.
    unique_execution: bool = True

    @classmethod
    def _get_candidates(cls, n_loops: int) -> list[list[int]]:
        """Return all non-identity permutations, capped at MAX_VOCAB_SIZE_PER_SLOT."""
        identity = list(range(n_loops))
        candidates = [list(p) for p in permutations(identity) if list(p) != identity]
        return candidates[:MAX_VOCAB_SIZE_PER_SLOT]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "New loop iterator order as a list of dimension indices.",
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
        # Must be a valid permutation
        if sorted(perm) != list(range(n)):
            return False
        # Must not be identity
        if perm == list(range(n)):
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
            f'    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %generic iterator_interchange = {perm} : (!transform.any_op) -> !transform.any_op\n'
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
        if 'tag = "operation_0"' not in after:
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
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        candidates = cls._get_candidates(n_loops)
        if not candidates:
            return {"permutation": list(range(n_loops))}
        idx = raw_slots[0] % len(candidates)
        return {"permutation": candidates[idx]}
