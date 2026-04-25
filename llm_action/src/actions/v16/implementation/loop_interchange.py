from itertools import permutations

from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """
    Loop Interchange action: permutes the loop ordering of a linalg operation
    using transform.structured.generalize + transform.structured.interchange.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "permutation": {
                "description": "New ordering of loop iterators (non-identity permutation)",
                "type": "list[int]",
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        perm = params.get("permutation", [])
        if not perm:
            return False
        # Must be a valid permutation
        n = len(perm)
        if sorted(perm) != list(range(n)):
            return False
        # Must be non-identity
        if perm == list(range(n)):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        perm = params["permutation"]
        n_iterators = cls._count_iterators(code)

        # Extend permutation to full iterator count if needed
        full_perm = list(perm)
        if len(full_perm) < n_iterators:
            # Keep remaining dims in original order
            used = set(full_perm)
            for i in range(n_iterators):
                if i not in used:
                    full_perm.append(i)
        full_perm = full_perm[:n_iterators]

        perm_str = str(full_perm)

        transform_code = (
            f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %generic = transform.structured.generalize %op'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %generic'
            f' iterator_interchange = {perm_str}'
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
        return [len(candidates)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        candidates = cls._get_candidates(n_loops)
        idx = raw_slots[0] % len(candidates)
        return {"permutation": candidates[idx]}

    @classmethod
    def _get_candidates(cls, n_loops: int) -> list[list[int]]:
        """Generate non-identity permutations, capped at MAX_VOCAB_SIZE_PER_SLOT."""
        identity = list(range(n_loops))
        candidates = []
        for p in permutations(range(n_loops)):
            p_list = list(p)
            if p_list != identity:
                candidates.append(p_list)
                if len(candidates) >= MAX_VOCAB_SIZE_PER_SLOT:
                    break
        if not candidates:
            candidates = [identity]
        return candidates

    @staticmethod
    def _count_iterators(code: str) -> int:
        import re
        match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
        if match:
            return len(match.group(1).split(","))
        if "conv_2d_nchw_fchw" in code:
            return 7
        if "matmul" in code:
            return 3
        return 7
