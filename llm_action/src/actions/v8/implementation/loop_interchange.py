from itertools import permutations

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """
    Reorder (permute) the loops in a loop nest to improve memory access
    patterns, enable vectorization of the optimal dimension, or expose
    parallelism in outer loops.

    Uses transform.structured.generalize (for named ops) followed by
    transform.structured.interchange with iterator_interchange.
    """

    @classmethod
    def _get_candidates(cls, n_loops: int) -> list[list[int]]:
        """Generate adjacent swap permutations for n_loops dimensions."""
        identity = list(range(n_loops))
        candidates = []
        for i in range(n_loops - 1):
            perm = list(identity)
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
            candidates.append(perm)
        # Add a few more useful permutations for small n_loops
        if n_loops >= 3:
            # Reverse
            candidates.append(list(reversed(identity)))
            # Move last to first
            perm = [identity[-1]] + identity[:-1]
            candidates.append(perm)
            # Move first to last
            perm = identity[1:] + [identity[0]]
            candidates.append(perm)
        # Remove duplicates
        seen = set()
        unique = []
        for c in candidates:
            key = tuple(c)
            if key not in seen and key != tuple(identity):
                seen.add(key)
                unique.append(c)
        return unique if unique else [list(identity)]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "Permutation of loop indices (0-based). Must be a valid permutation.",
                "type": "list[int]",
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        permutation = params.get("permutation")
        if not permutation or not isinstance(permutation, list):
            return False
        n = len(permutation)
        if n < 2:
            return False
        if sorted(permutation) != list(range(n)):
            return False
        # Reject identity permutation
        if permutation == list(range(n)):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        permutation = params["permutation"]

        # Generalize first (converts named linalg ops to linalg.generic),
        # then interchange. Already-generic ops pass through generalize as no-op.
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %generic = transform.structured.generalize %op'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %generic'
            f' iterator_interchange = {str(permutation)}'
            f' : (!transform.any_op) -> !transform.any_op\n'
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
        return [len(candidates)] + [1] * (MAX_PARAM_SLOTS - 1)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        candidates = cls._get_candidates(n_loops)
        idx = raw_slots[0] % len(candidates)
        return {"permutation": candidates[idx]}
