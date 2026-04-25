from itertools import permutations

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """
    Permute the loops of the target linalg op's iteration space via
    `transform.structured.interchange`. The target op is first generalized to
    `linalg.generic` because interchange only applies to generic ops.

    Parameters:
      - permutation: list[int], a permutation of range(n_loops) describing the
        new loop order.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": (
                    "Permutation of loop indices describing the new loop order "
                    "for the target linalg op's iteration space."
                ),
                "type": "list[int]",
            }
        }

    @staticmethod
    def _candidates(n_loops: int) -> list[list[int]]:
        n = max(int(n_loops), 1)
        identity = tuple(range(n))
        perms = [list(p) for p in permutations(range(n)) if tuple(p) != identity]
        # Cap to MAX_VOCAB_SIZE_PER_SLOT to fit within the RL policy budget.
        return perms[:MAX_VOCAB_SIZE_PER_SLOT] if perms else [list(range(n))]

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        perm = params.get("permutation")
        if not isinstance(perm, (list, tuple)) or len(perm) == 0:
            return False
        n = len(perm)
        if sorted(perm) != list(range(n)):
            return False
        # Reject identity: no-op.
        if list(perm) == list(range(n)):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        perm = list(params["permutation"])
        perm_str = "[" + ", ".join(str(int(p)) for p in perm) + "]"

        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op
    %interchanged = transform.structured.interchange %generic iterator_interchange = {perm_str} : (!transform.any_op) -> !transform.any_op
    %tag = transform.param.constant "operation_0" -> !transform.any_param
    transform.annotate %interchanged "tag" = %tag : !transform.any_op, !transform.any_param
    transform.yield
  }}
}}
"""
        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        if not after or "func.func" not in after:
            return False
        if after.strip() == before.strip():
            return False
        if 'tag = "operation_0"' not in after:
            return False
        # Expect a generic op to be present after generalization + interchange.
        return "linalg.generic" in after

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls._candidates(n_loops))]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        candidates = cls._candidates(n_loops)
        idx = raw_slots[0] % len(candidates)
        return {"permutation": candidates[idx]}
