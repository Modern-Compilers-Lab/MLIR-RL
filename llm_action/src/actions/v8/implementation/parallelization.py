from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS
from llm_action.src.utils.transformation import run_transform_code


class Parallelization(ActionBase):
    """
    Distribute iterations of parallel (non-reduction) loops across multiple
    CPU threads for concurrent execution on available cores.
    Uses transform.structured.tile_using_forall with num_threads to create
    an scf.forall parallel region around the tagged operation.
    """

    THREAD_OPTIONS = [2, 4, 7, 8, 14, 28]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "num_threads": {
                "description": "Number of threads to distribute the first parallel dimension across.",
                "type": "int",
                "values": cls.THREAD_OPTIONS,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        num_threads = params.get("num_threads")
        if not isinstance(num_threads, int) or num_threads < 2:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        num_threads = params["num_threads"]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %tiled_op:2 = transform.structured.tile_using_forall %op'
            f' num_threads [{num_threads}]'
            f' : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %tiled_op#0 "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return [len(cls.THREAD_OPTIONS)] + [1] * (MAX_PARAM_SLOTS - 1)

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {"num_threads": cls.THREAD_OPTIONS[raw_slots[0] % len(cls.THREAD_OPTIONS)]}
