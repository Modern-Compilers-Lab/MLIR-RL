from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class LoopDistribution(ActionBase):
    """
    Split a linalg operation along a chosen dimension at a specified chunk
    size boundary, producing two complementary operations that together
    cover the original iteration domain.
    """

    DIM_OPTIONS = [0, 1, 2]
    CHUNK_OPTIONS = [4, 8, 16, 32, 64]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "dimension": {
                "description": "Which loop dimension to split (0-indexed).",
                "type": "int",
                "values": cls.DIM_OPTIONS,
            },
            "chunk_size": {
                "description": "Split point (size of the lower half).",
                "type": "int",
                "values": cls.CHUNK_OPTIONS,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        dim = params.get("dimension", -1)
        chunk = params.get("chunk_size", 0)
        if dim < 0 or chunk <= 0:
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        dim = params["dimension"]
        chunk = params["chunk_size"]

        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %split = transform.structured.split %op after {chunk}'
            f' {{ dimension = {dim} }}'
            f' : !transform.any_op\n'
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
        return 2

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return [min(n_loops, len(cls.DIM_OPTIONS)), len(cls.CHUNK_OPTIONS)]

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        n_dims = min(n_loops, len(cls.DIM_OPTIONS))
        dim = raw_slots[0] % n_dims
        chunk = cls.CHUNK_OPTIONS[raw_slots[1] % len(cls.CHUNK_OPTIONS)]
        return {"dimension": dim, "chunk_size": chunk}
