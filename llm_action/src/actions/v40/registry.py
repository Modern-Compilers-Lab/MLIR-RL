from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v40.implementation.tiling import Tiling
from llm_action.src.actions.v40.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v40.implementation.vectorization import Vectorization
from llm_action.src.actions.v40.implementation.unrolling import Unrolling
from llm_action.src.actions.v40.implementation.parallelization import Parallelization
from llm_action.src.actions.v40.implementation.promotion import Promotion

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Vectorization,
    # Unrolling,
    Parallelization,
    # Promotion,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Vectorization": [
        "Tiling",
        "LoopInterchange",
        "Vectorization",
        # "Unrolling",
        "Parallelization",
        # "Promotion",
    ],
}
