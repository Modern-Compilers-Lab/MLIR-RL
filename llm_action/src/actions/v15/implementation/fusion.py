from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Fusion(ActionBase):
    """
    Tile the target operation, generalize the inner operation to
    linalg.generic form, and interchange the iterator dimensions so that
    the reduction dimension is moved innermost for better data reuse.
    Combines tiling, generalization, and loop reordering in a single action.
    """

    VOCAB = [0, 4, 8, 16, 32]

    # Pre-computed permutations for 3 iterators (matmul: M, N, K)
    # Move the last parallel dimension before the first, keeping reduction last
    INTERCHANGE_3 = [1, 0, 2]  # (N, M, K) — swap M and N

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile sizes for the fusion region. 0 means no tiling/fusing.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        tile_sizes = params.get("tile_sizes", [])
        if not tile_sizes or all(s == 0 for s in tile_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = params["tile_sizes"]
        n_loops = sum(1 for s in tile_sizes if s != 0)
        if n_loops == 0:
            return code

        loop_types = ", ".join(["!transform.any_op"] * n_loops)
        tile_sizes_str = str(tile_sizes).replace("'", "")

        # Build the interchange permutation based on the number of iterators
        n_iters = len(tile_sizes)
        if n_iters >= 3:
            perm = cls.INTERCHANGE_3[:n_iters]
            interchange_line = (
                f'    %interchanged = transform.structured.interchange %generic'
                f' iterator_interchange = {perm}'
                f' : (!transform.any_op) -> !transform.any_op\n'
            )
            tag_target = "%interchanged"
        else:
            # For 2 or fewer iterators, skip interchange
            interchange_line = ""
            tag_target = "%generic"

        if n_loops > 1:
            tile_result = (
                f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op'
                f' tile_sizes {tile_sizes_str}'
                f' : (!transform.any_op) -> (!transform.any_op, {loop_types})\n'
            )
        else:
            tile_result = (
                f'    %tiled_op, %loops = transform.structured.tile_using_for %op'
                f' tile_sizes {tile_sizes_str}'
                f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            )

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            + tile_result +
            f'    %generic = transform.structured.generalize %tiled_op'
            f' : (!transform.any_op) -> !transform.any_op\n'
            + interchange_line +
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate {tag_target} "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return MAX_PARAM_SLOTS

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [len(cls.VOCAB)] * min(n_loops, MAX_PARAM_SLOTS)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n = min(n_loops, MAX_PARAM_SLOTS)
        sizes = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"tile_sizes": sizes}
