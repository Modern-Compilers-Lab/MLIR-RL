from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class ParallelizationDirect(ActionBase):
    """Directly partition iterations of parallel loops across a fixed
    number of threads using scf.forall with num_threads.

    Single-shot: introduces scf.forall which changes the loop kind;
    a second application would try to parallelize already-parallel loops."""

    unique_execution: bool = True  # introduces forall, changes loop structure

    # Thread counts that divide common batch/filter sizes (128, 256, etc.)
    VOCAB = [0, 2, 4, 7, 14]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "num_threads": {
                "description": "Number of threads per parallel loop dimension (0 = skip); reduction dims are automatically zeroed",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        num_threads = params.get("num_threads", [])
        if not num_threads or all(t == 0 for t in num_threads):
            return False
        return True

    @classmethod
    def _extract_parallel_dims(cls, code: str) -> list[bool]:
        """Determine which dims are parallel."""
        if "linalg.conv_2d_nchw_fchw" in code:
            return [True, True, True, True, False, False, False]
        if "iterator_types" in code:
            import re
            match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
            if match:
                types = match.group(1)
                return ["parallel" in t for t in types.split(",")]
        return [True] * 7

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        num_threads = list(params["num_threads"])
        parallel_dims = cls._extract_parallel_dims(code)

        # Zero out reduction dims
        for i in range(min(len(num_threads), len(parallel_dims))):
            if not parallel_dims[i]:
                num_threads[i] = 0

        if all(t == 0 for t in num_threads):
            return code

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op, %forall = transform.structured.tile_using_forall %op num_threads {num_threads} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
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
        threads = [cls.VOCAB[raw_slots[i] % len(cls.VOCAB)] for i in range(n)]
        return {"num_threads": threads}
