from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class ParallelizationTiling(ActionBase):
    """Tile outer parallel loops into chunks and distribute them across
    threads using scf.forall.

    Single-shot: introduces scf.forall which changes the loop kind;
    a second application would try to parallelize already-parallel loops."""

    unique_execution: bool = True  # introduces forall, changes loop structure

    VOCAB = [0, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "Tile size per parallel loop dimension for thread distribution (0 = skip); reduction dims are automatically zeroed",
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
    def _extract_parallel_dims(cls, code: str) -> list[bool]:
        """Determine which dims are parallel by checking iterator_types or
        inferring from the conv2d structure."""
        if "linalg.conv_2d_nchw_fchw" in code:
            # conv2d has 7 dims: [N, F, OH, OW, C, KH, KW]
            # First 4 are parallel, last 3 are reduction
            return [True, True, True, True, False, False, False]
        if "iterator_types" in code:
            import re
            match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
            if match:
                types = match.group(1)
                return ["parallel" in t for t in types.split(",")]
        # Default: assume all dims are parallel (conservative)
        return [True] * 7

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        tile_sizes = list(params["tile_sizes"])
        parallel_dims = cls._extract_parallel_dims(code)

        # Zero out reduction dims to avoid incorrect parallelization
        for i in range(min(len(tile_sizes), len(parallel_dims))):
            if not parallel_dims[i]:
                tile_sizes[i] = 0

        if all(s == 0 for s in tile_sizes):
            return code

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
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
