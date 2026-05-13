from itertools import permutations

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """
    Reorder loop dimensions in a loop nest to improve memory access stride
    patterns and enable downstream optimizations.
    Requires generalization before interchange (named linalg ops -> generic).
    """

    unique_execution: bool = False  # can interchange at different nesting levels after tiling

    @classmethod
    def _get_candidates(cls, n_loops: int) -> list[list[int]]:
        identity = list(range(n_loops))
        all_perms = [list(p) for p in permutations(range(n_loops)) if list(p) != identity]
        return all_perms[:MAX_VOCAB_SIZE_PER_SLOT] if all_perms else [identity]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "Reordering vector for loop dimensions (non-identity permutation).",
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
        if perm == list(range(len(perm))):
            return False
        if sorted(perm) != list(range(len(perm))):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        perm = params["permutation"]
        perm_str = str(perm)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %gen = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n'
            f'    %inter = transform.structured.interchange %gen iterator_interchange = {perm_str} : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %inter "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        candidates = cls._get_candidates(n_loops)
        idx = raw_slots[0] % len(candidates)
        return {"permutation": candidates[idx]}
