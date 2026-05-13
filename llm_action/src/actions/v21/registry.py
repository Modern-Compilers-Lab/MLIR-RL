from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v21.implementation.tiling import Tiling
from llm_action.src.actions.v21.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v21.implementation.promotion import Promotion
from llm_action.src.actions.v21.implementation.vectorization import Vectorization
from llm_action.src.actions.v21.implementation.unrolling import Unrolling
from llm_action.src.actions.v21.implementation.packing import Packing
from llm_action.src.actions.v21.implementation.parallelization import Parallelization
from llm_action.src.actions.v21.implementation.peeling import Peeling

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Vectorization,
    Unrolling,
    Packing,
    Parallelization,
    # Peeling,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Vectorization": ["Tiling", "LoopInterchange", "Promotion", "Vectorization", "Unrolling", "Packing", "Parallelization"],
    "Packing": ["LoopInterchange", "Vectorization", "Packing"],
    "Unrolling": ["Packing"],
    # "Peeling": ["Packing", "Peeling"],
}
