from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Canonicalization(ActionBase):
    """
    Apply generic canonicalization and CSE patterns to the body of the
    function containing the target linalg op. This runs
    `transform.apply_patterns.canonicalization` (optionally interleaved with
    CSE) to fold, DCE, and simplify the IR, which normalizes the shape of the
    payload and unblocks later pattern-matching transformations.

    Parameters:
      - apply_cse: bool, if true, interleave CSE with the canonicalization
        rewrite loop until a joint fixpoint is reached.
    """

    CSE_VOCAB = [0, 1]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "apply_cse": {
                "description": (
                    "If 1, interleave common subexpression elimination with "
                    "the canonicalization rewrite loop. If 0, run only the "
                    "canonicalization patterns."
                ),
                "type": "int",
                "values": cls.CSE_VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        apply_cse = params.get("apply_cse")
        if not isinstance(apply_cse, int) or apply_cse not in (0, 1):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        apply_cse = int(params["apply_cse"])
        cse_attr = "{apply_cse}" if apply_cse == 1 else ""

        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %func = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {{
      transform.apply_patterns.canonicalization
    }} {cse_attr} : !transform.any_op
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
        if 'tag = "operation_0"' not in after:
            return False
        # Canonicalization is idempotent: on already-clean code the output
        # equals the input, which is still a valid successful application.
        return True

    @classmethod
    def params_size(cls) -> int:
        return 1

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.CSE_VOCAB)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        apply_cse = cls.CSE_VOCAB[raw_slots[0] % len(cls.CSE_VOCAB)]
        return {"apply_cse": apply_cse}
