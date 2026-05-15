from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Vectorization(ActionBase):
    """Tile to SIMD-friendly sizes and generalize pooling to linalg.generic.

    Pooling_nchw_max has windowed access patterns (oh*stride+kh, ow*stride+kw)
    that prevent direct MLIR-level vectorization. This action tiles the parallel
    output dimensions to AVX2-friendly widths and generalizes the named op to
    linalg.generic, enabling the LLVM backend to auto-vectorize the resulting
    loop nest with optimal register usage.

    Category B (lowering): the named pooling op is consumed and replaced by
    a linalg.generic. A second application would find a generic op (not pooling)
    and generalization is a no-op on already-generic ops, so unique_execution=True.
    """

    # Generalizes pooling to generic (one-way lowering).
    unique_execution: bool = True

    VOCAB = [1, 2, 4, 8, 16]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "vector_sizes": {
                "description": "SIMD-friendly tile sizes per loop dimension. "
                "Non-1 entries select dimensions to vectorize at AVX2 widths.",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        vsizes = params.get("vector_sizes", [])
        if not vsizes or not isinstance(vsizes, list):
            return False
        if all(s <= 1 for s in vsizes):
            return False
        for s in vsizes:
            if s < 0:
                return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        vsizes = params["vector_sizes"]

        # Build tile sizes: only tile dimensions with vector_size > 1.
        # Use 0 for dimensions that don't need tiling (size 1 or size matches dim).
        tile_sizes = [s if s > 1 else 0 for s in vsizes]

        n_loops = sum(1 for s in tile_sizes if s != 0)

        if n_loops == 0:
            return code

        loop_results = ", ".join(["!transform.any_op"] * n_loops)

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %loops:{n_loops} = transform.structured.tile_using_for %op tile_sizes {tile_sizes}'
            f' : (!transform.any_op) -> (!transform.any_op, {loop_results})\n'
            f'    %gen = transform.structured.generalize %tiled_op : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %gen "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return {"vector_sizes": sizes}
