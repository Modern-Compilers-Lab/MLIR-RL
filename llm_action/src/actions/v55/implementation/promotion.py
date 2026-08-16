from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Promotion(ActionBase):
    """Copy tiled operand data into contiguous temporary local buffers (alloca).

    Promotion requires tiling + bufferization as internal preprocessing.
    The action tiles the first 2 dims with size 32, bufferizes, then promotes.
    """

    unique_execution: bool = False  # can promote different operand subsets

    OPERAND_SETS = [
        [0, 1, 2],  # all operands
        [0, 1],     # both inputs
        [0],        # first input only
        [1],        # second input only
        [2],        # output only
        [0, 2],     # first input + output
    ]

    TILE_SIZE = 32  # fixed tile size for internal preprocessing

    @classmethod
    def parameters(cls) -> dict:
        return {
            "operands_to_promote": {
                "description": "List of operand indices to copy into contiguous local buffers.",
                "type": "list[int]",
                "values": cls.OPERAND_SETS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        ops_to_promote = params.get("operands_to_promote", [])
        if not ops_to_promote or not isinstance(ops_to_promote, list):
            return False
        if any(not isinstance(o, int) or o < 0 for o in ops_to_promote):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        ops_to_promote = params["operands_to_promote"]
        ts = cls.TILE_SIZE

        # Count non-zero tile sizes (we tile first 2 dims)
        n_tile_loops = 2
        loop_results = ", ".join(["!transform.any_op"] * n_tile_loops)

        promote_attr = f"{{operands_to_promote = {ops_to_promote}, use_alloca}}"

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%module: !transform.any_op {{transform.consumed}}) {{\n'
            f'    %op0 = transform.structured.match attributes{{tag = "operation_0"}} in %module : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_tile_loops} = transform.structured.tile_using_for %op0 tile_sizes [{ts}, {ts}] : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    %tiled_tag = transform.param.constant "tiled_target" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tiled_tag : !transform.any_op, !transform.any_param\n'
            f'\n'
            f'    %buf = transform.bufferization.one_shot_bufferize layout{{IdentityLayoutMap}} %module {{bufferize_function_boundaries = true}} : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    %func1 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted_target = transform.structured.match attributes{{tag = "tiled_target"}} in %func1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %promoted_op = transform.structured.promote %promoted_target {promote_attr} : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    %func2 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    transform.apply_registered_pass "canonicalize" to %func2 : (!transform.any_op) -> !transform.any_op\n'
            f'\n'
            f'    %func3 = transform.structured.match ops{{["func.func"]}} attributes{{sym_name = "main"}} in %buf : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_op = transform.structured.match attributes{{tag = "tiled_target"}} in %func3 : (!transform.any_op) -> !transform.any_op\n'
            f'    %final_tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %final_op "tag" = %final_tag : !transform.any_op, !transform.any_param\n'
            f'\n'
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
        return [len(cls.OPERAND_SETS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int, loop_bounds: list[int] | None = None) -> dict:
        idx = raw_slots[0] % len(cls.OPERAND_SETS)
        return {"operands_to_promote": cls.OPERAND_SETS[idx]}
