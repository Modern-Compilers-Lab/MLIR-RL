from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v24.implementation.tiling import Tiling
from llm_action.src.actions.v24.implementation.promotion import Promotion
from llm_action.src.actions.v24.implementation.packing import Packing
from llm_action.src.actions.v24.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v24.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v24.implementation.loop_peeling import LoopPeeling
from llm_action.src.actions.v24.implementation.vectorization import Vectorization
from llm_action.src.actions.v24.implementation.parallelization import Parallelization
from llm_action.src.actions.v24.implementation.canonicalization import Canonicalization

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    Promotion,
    Packing,
    LoopInterchange,
    LoopUnrolling,
    LoopPeeling,
    Vectorization,
    Parallelization,
    Canonicalization,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    # Vectorization replaces linalg.matmul with vector ops; tag moves to scf.for,
    # all linalg-targeting actions lose their payload op
    "Vectorization": [
        "Tiling",
        "Promotion",
        "Packing",
        "LoopInterchange",
        "LoopUnrolling",
        "LoopPeeling",
        "Parallelization",
    ],
    # Promotion bufferizes tensors to memrefs; Packing requires tensor IR,
    # Canonicalization is a no-op on bufferized form
    "Promotion": ["Packing", "Canonicalization"],
    # Packing reshapes linalg.matmul into 6-dim linalg.generic;
    # Vectorization precondition rejects the packed form
    "Packing": ["Vectorization"],
}
