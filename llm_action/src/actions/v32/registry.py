from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v32.implementation.tiling import Tiling
from llm_action.src.actions.v32.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v32.implementation.promotion import Promotion
from llm_action.src.actions.v32.implementation.vectorization import Vectorization
from llm_action.src.actions.v32.implementation.packing import Packing
from llm_action.src.actions.v32.implementation.parallelization import Parallelization
from llm_action.src.actions.v32.implementation.parallel_tiling import ParallelTiling

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Vectorization,
    Packing,
    Parallelization,
    ParallelTiling,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Vectorization": ["Tiling", "LoopInterchange", "Promotion", "Vectorization", "Packing", "Parallelization", "ParallelTiling"],
    "Packing": ["Vectorization", "Promotion"],
}
