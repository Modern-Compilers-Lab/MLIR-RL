from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Peeling(ActionBase):
    """Peels remainder iterations into a distinct epilogue loop using transform.loop.peel.

    Separates the leftover (remainder) iterations of an scf.for loop into a
    separate loop, enabling the main loop body to assume evenly divisible trip
    counts.  This is most useful after tiling, where the tile size may not
    evenly divide the iteration domain.

    Requires that the code already contains scf.for loops (i.e. a prior tiling
    action has been applied).
    """

    PEEL_OPTIONS = [False, True]  # peel back (False) or peel front (True)

    # ------------------------------------------------------------------
    # ActionBase interface
    # ------------------------------------------------------------------

    @classmethod
    def parameters(cls) -> dict:
        return {
            "peel_front": {
                "description": (
                    "Whether to peel iterations from the front of the loop "
                    "(True) or the back (False)."
                ),
                "type": "bool",
                "values": cls.PEEL_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        # Tag must exist in the code.
        if 'tag = "operation_0"' not in code:
            return False
        # Code must already contain scf.for loops (from a prior tiling step).
        if "scf.for" not in code:
            return False
        peel_front = params.get("peel_front")
        if not isinstance(peel_front, bool):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        peel_front = params["peel_front"]
        peel_front_str = "true" if peel_front else "false"

        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %parent_loop = transform.get_parent_op %op {{op_name = "scf.for"}} : (!transform.any_op) -> !transform.op<"scf.for">\n'
            f"    %peeled, %remainder = transform.loop.peel %parent_loop {{peel_front = {peel_front_str}, fail_if_already_divisible = false}} : (!transform.op<\"scf.for\">) -> (!transform.op<\"scf.for\">, !transform.op<\"scf.for\">)\n"
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
        return [len(cls.PEEL_OPTIONS)]  # 2: binary choice

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        peel_front = cls.PEEL_OPTIONS[raw_slots[0] % len(cls.PEEL_OPTIONS)]
        return {"peel_front": peel_front}
