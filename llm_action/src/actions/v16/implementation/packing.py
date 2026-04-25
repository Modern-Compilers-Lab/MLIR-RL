from llm_action.src.actions.base import ActionBase
from llm_action.src.config import MAX_PARAM_SLOTS, MAX_VOCAB_SIZE_PER_SLOT
from llm_action.src.utils.transformation import run_transform_code


class Packing(ActionBase):
    """
    Packing action: reorganizes data into contiguous packed buffers
    using transform.structured.pack, inserting linalg.pack/unpack ops.
    """

    VOCAB = [0, 4, 8, 16, 32]

    @classmethod
    def parameters(cls) -> dict:
        return {
            "packed_sizes": {
                "description": "Pack sizes per iterator dimension (0 = do not pack)",
                "type": "list[int]",
                "values": cls.VOCAB,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        if 'tag = "operation_0"' not in code:
            return False
        packed_sizes = params.get("packed_sizes", [])
        if not packed_sizes or all(s == 0 for s in packed_sizes):
            return False
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        packed_sizes = params["packed_sizes"]

        n_iterators = cls._count_iterators(code)
        full_sizes = list(packed_sizes) + [0] * max(0, n_iterators - len(packed_sizes))
        full_sizes = full_sizes[:n_iterators]

        if all(s == 0 for s in full_sizes):
            return code

        transform_code = (
            f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1'
            f' : (!transform.any_op) -> !transform.any_op\n'
            f'    %packed = transform.structured.pack %op'
            f' packed_sizes = {str(full_sizes)} : (!transform.any_op) -> !transform.any_op\n'
            f'    %tag = transform.param.constant "operation_0" -> !transform.any_param\n'
            f'    transform.annotate %packed "tag" = %tag : !transform.any_op, !transform.any_param\n'
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
        return {"packed_sizes": sizes}

    @staticmethod
    def _count_iterators(code: str) -> int:
        import re
        match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
        if match:
            return len(match.group(1).split(","))
        if "conv_2d_nchw_fchw" in code:
            return 7
        if "matmul" in code:
            return 3
        return 7
