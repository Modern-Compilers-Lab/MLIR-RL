from itertools import permutations

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """Permute loop dimensions via generalize + interchange.

    Repeatable: can interchange at different nesting levels after tiling.
    """

    unique_execution: bool = True  # different permutations at different nesting levels

    @classmethod
    def _get_candidates(cls, n_loops: int) -> list[list[int]]:
        identity = list(range(n_loops))
        all_perms = [list(p) for p in permutations(range(n_loops)) if list(p) != identity]
        return all_perms[:MAX_VOCAB_SIZE_PER_SLOT]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": "Loop dimension permutation. Must be a valid non-identity permutation of [0..n_loops-1]."
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        perm = params.get("permutation", [])
        if not perm:
            return False
        if perm == list(range(len(perm))):
            return False  # identity = no-op
        if sorted(perm) != list(range(len(perm))):
            return False  # not a valid permutation
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        perm = params["permutation"]
        perm_str = str(perm)

        # Generalize first (interchange only works on linalg.generic, not named ops)
        # Then interchange with the given permutation
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n"
            f"    %interchanged = transform.structured.interchange %generic iterator_interchange = {perm_str} : (!transform.any_op) -> !transform.any_op\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f"    transform.annotate %interchanged \"tag\" = %tag : !transform.any_op, !transform.any_param\n"
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
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        candidates = cls._get_candidates(n_loops)
        return [max(len(candidates), 1)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        candidates = cls._get_candidates(n_loops)
        if not candidates:
            return {"permutation": list(range(n_loops))}
        idx = raw_slots[0] % len(candidates)
        return {"permutation": candidates[idx]}
