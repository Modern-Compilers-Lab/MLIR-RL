from itertools import permutations

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """Reorders iterator dimensions of a linalg operation using generalize + interchange.

    Generalizes the tagged operation first (converting named ops like matmul
    to linalg.generic), then applies an interchange permutation to reorder
    the iteration dimensions.  A non-identity permutation can improve spatial
    locality or enable downstream vectorization by placing a stride-1
    dimension innermost.
    """

    @staticmethod
    def _get_candidates(n_loops: int) -> list[list[int]]:
        """Generate all non-identity permutations for up to MAX_PARAM_SLOTS iterators.

        Args:
            n_loops: Number of iterator dimensions in the target operation.

        Returns:
            List of permutations (each a list[int]), excluding the identity,
            capped at MAX_VOCAB_SIZE_PER_SLOT entries.
        """
        effective = min(n_loops, MAX_PARAM_SLOTS)  # MAX_PARAM_SLOTS == 3
        identity = list(range(effective))
        candidates = [
            list(p) for p in permutations(range(effective)) if list(p) != identity
        ]
        return candidates[:MAX_VOCAB_SIZE_PER_SLOT]

    # ------------------------------------------------------------------
    # ActionBase interface
    # ------------------------------------------------------------------

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "Permutation of iterator indices (0-based).",
                "type": "list[int]",
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        # Tag must exist in the code.
        if 'tag = "operation_0"' not in code:
            return False

        permutation = params.get("permutation")
        if not permutation or not isinstance(permutation, list):
            return False

        n = len(permutation)

        # Must be a valid permutation of [0 .. n-1].
        if sorted(permutation) != list(range(n)):
            return False

        # Identity permutation is a no-op -- reject it.
        if permutation == list(range(n)):
            return False

        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        permutation = params["permutation"]
        perm_list = ", ".join(str(p) for p in permutation)

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n"
            f"    %interchanged = transform.structured.interchange %generic iterator_interchange = [{perm_list}] : (!transform.any_op) -> !transform.any_op\n"
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

    # ------------------------------------------------------------------
    # RL parameter interface
    # ------------------------------------------------------------------

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        candidates = cls._get_candidates(n_loops)
        return [max(1, len(candidates))]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        candidates = cls._get_candidates(n_loops)
        if not candidates:
            return {"permutation": list(range(n_loops))}
        idx = raw_slots[0] % len(candidates)
        return {"permutation": candidates[idx]}
