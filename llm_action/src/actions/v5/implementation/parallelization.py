from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Parallelization(ActionBase):
    """
    Distribute iterations of a tagged linalg operation across threads
    using scf.forall. Tiles the operation with tile_using_forall to
    create parallel work partitions.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "List of tile sizes for parallel dimensions. 0 means do not parallelize that dimension.",
                "type": "list[int]",
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not isinstance(tile_sizes, list) or len(tile_sizes) == 0:
            return False
        if not all(isinstance(s, int) and s >= 0 for s in tile_sizes):
            return False
        if all(s == 0 for s in tile_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg0 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall = transform.structured.tile_using_forall %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
