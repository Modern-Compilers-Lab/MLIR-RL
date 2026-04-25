from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopUnrolling(ActionBase):
    """Unrolls the innermost scf.for loop around a tagged operation.

    Replicates the loop body multiple times per iteration to reduce loop
    overhead (branch, counter increment) and expose instruction-level
    parallelism (ILP).  Operates on code that ALREADY contains scf.for
    loops from a previous tiling action.

    The transform finds the tagged linalg operation, obtains its nearest
    parent scf.for loop via ``transform.get_parent_op``, and unrolls it
    with ``transform.loop.unroll``.  The unroll transform does not return
    a handle (the loop may be fully removed), but the tagged operation
    inside the loop body survives naturally since unrolling just
    duplicates the body.
    """

    UNROLL_OPTIONS = [2, 3, 4, 8, 16]

    # ------------------------------------------------------------------
    # ActionBase interface
    # ------------------------------------------------------------------

    @classmethod
    def parameters(cls) -> dict:
        return {
            "unroll_factor": {
                "description": "Number of times to replicate the loop body per iteration.",
                "type": "int",
                "values": cls.UNROLL_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        # Tag must exist in the code.
        if 'tag = "operation_0"' not in code:
            return False
        # Code must already have scf.for loops (from a previous tiling action).
        if "scf.for" not in code:
            return False
        unroll_factor = params.get("unroll_factor")
        if not isinstance(unroll_factor, int) or unroll_factor < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        unroll_factor = params["unroll_factor"]

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %parent_loop = transform.get_parent_op %op {{op_name = "scf.for"}} : (!transform.any_op) -> !transform.op<"scf.for">\n'
            f"    transform.loop.unroll %parent_loop {{factor = {unroll_factor}}} : !transform.op<\"scf.for\">\n"
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
        return [len(cls.UNROLL_OPTIONS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        unroll_factor = cls.UNROLL_OPTIONS[raw_slots[0] % len(cls.UNROLL_OPTIONS)]
        return {"unroll_factor": unroll_factor}
