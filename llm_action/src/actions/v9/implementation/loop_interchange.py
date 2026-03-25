from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """Reorder (permute) loops in a loop nest to change which dimension
    is innermost, affecting memory access stride patterns and enabling
    contiguous vector loads/stores. Generalizes named ops first."""

    TAG = 'tag = "operation_0"'

    @classmethod
    def _get_candidates(cls, n_loops: int) -> list[list[int]]:
        """Generate all pair-swap permutations, capped at MAX_VOCAB_SIZE_PER_SLOT."""
        candidates = []
        for i in range(n_loops):
            for j in range(i + 1, n_loops):
                perm = list(range(n_loops))
                perm[i], perm[j] = perm[j], perm[i]
                candidates.append(perm)
        return candidates[:MAX_VOCAB_SIZE_PER_SLOT]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "Permutation of loop indices (pair swap)",
                "type": "list[int]",
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if cls.TAG not in code:
            return False
        permutation = params.get("permutation", [])
        if not permutation or len(permutation) < 2:
            return False
        # Check it's a valid permutation
        if sorted(permutation) != list(range(len(permutation))):
            return False
        # Reject identity permutation (no-op)
        if permutation == list(range(len(permutation))):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """Generalize named linalg ops to linalg.generic (standalone)."""
        transform_code = (
            'module attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {\n'
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            '    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n'
            '    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            '    transform.annotate %generic "tag" = %tag : !transform.any_op, !transform.any_param\n'
            '    transform.yield\n'
            '  }\n'
            '}\n'
        )
        try:
            return run_transform_code(code, transform_code)
        except Exception:
            return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        permutation = params["permutation"]

        # Combined generalize + interchange in a single transform pass
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %generic = transform.structured.generalize %op : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %generic iterator_interchange = {permutation} : (!transform.any_op) -> !transform.any_op\n'
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
        return [max(len(candidates), 1)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        candidates = cls._get_candidates(n_loops)
        if not candidates:
            return {"permutation": list(range(n_loops))}
        idx = raw_slots[0] % len(candidates)
        return {"permutation": candidates[idx]}
