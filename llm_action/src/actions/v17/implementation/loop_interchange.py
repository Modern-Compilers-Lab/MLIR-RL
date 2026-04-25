from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """
    Reorders loop iterators of the tagged linalg operation via
    generalize + interchange. Changes loop ordering for better
    spatial locality and prefetching.
    """

    @classmethod
    def _get_candidates(cls, n_loops: int) -> list[list[int]]:
        """Generate non-identity adjacent-swap permutations, capped at MAX_VOCAB_SIZE_PER_SLOT."""
        identity = list(range(n_loops))
        candidates = []
        for i in range(n_loops - 1):
            perm = list(identity)
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
            candidates.append(perm)
            if len(candidates) >= MAX_VOCAB_SIZE_PER_SLOT:
                break
        return candidates

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "Permutation of iterator indices for the loop nest.",
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
        if len(perm) < 2:
            return False
        if sorted(perm) != list(range(len(perm))):
            return False
        if perm == list(range(len(perm))):
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
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f" : (!transform.any_op) -> !transform.any_op\n"
            f"    %generic = transform.structured.generalize %op"
            f" : (!transform.any_op) -> !transform.any_op\n"
            f"    %interchanged = transform.structured.interchange %generic"
            f" iterator_interchange = {perm_str}"
            f" : (!transform.any_op) -> !transform.any_op\n"
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %interchanged "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
