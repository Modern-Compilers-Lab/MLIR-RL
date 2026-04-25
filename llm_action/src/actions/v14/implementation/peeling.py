from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Peeling(ActionBase):
    """
    Peel the innermost tiled loop of the target linalg op so that its step
    evenly divides its range. Because the payload is initially a single
    structured linalg op, we first tile with `transform.structured.tile_using_for`
    to materialize an `scf.for` band and then apply `transform.loop.peel` on
    the innermost loop.

    Parameters:
      - tile_sizes: list[int], per-loop tile sizes used to materialize the
        `scf.for` band whose innermost loop is peeled.
    """

    VOCAB = [8, 16, 24, 48, 96]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": (
                    "Per-loop tile sizes used to create the scf.for band "
                    "whose innermost loop is peeled."
                ),
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        sizes = params.get("tile_sizes")
        if not isinstance(sizes, (list, tuple)) or len(sizes) == 0:
            return False
        if not all(isinstance(s, int) and s > 0 for s in sizes):
            return False
        # At least one loop must have a non-trivial tile to actually peel.
        if not any(s > 1 for s in sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        sizes = list(params["tile_sizes"])
        n_loops = len(sizes)
        result_types = ", ".join(["!transform.any_op"] * n_loops)
        sizes_str = "[" + ", ".join(str(int(s)) for s in sizes) + "]"
        # Peel the deepest loop with a non-trivial tile so the innermost
        # structured band is cleanly divisible by the step.
        inner_idx = max(
            (i for i, s in enumerate(sizes) if s > 1), default=n_loops - 1
        )

        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {sizes_str} : (!transform.any_op) -> (!transform.any_op, {result_types})
    %tag = transform.param.constant "operation_0" -> !transform.any_param
    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param
    %inner_loop = transform.cast %loops#{inner_idx} : !transform.any_op to !transform.op<"scf.for">
    %main, %remainder = transform.loop.peel %inner_loop : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">)
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
        if after.strip() == before.strip():
            return False
        if 'tag = "operation_0"' not in after:
            return False
        # Peeling replicates the innermost loop body into main and remainder;
        # we expect more scf.for ops than before.
        return after.count("scf.for") > before.count("scf.for")

    @classmethod
    def params_size(cls) -> int:
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(max(n_loops, 1), MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(max(n_loops, 1), MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"tile_sizes": sizes}
