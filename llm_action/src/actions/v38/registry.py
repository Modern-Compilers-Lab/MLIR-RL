from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v38.implementation.tiling import Tiling
from llm_action.src.actions.v38.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v38.implementation.promotion import Promotion
from llm_action.src.actions.v38.implementation.vectorization import Vectorization
from llm_action.src.actions.v38.implementation.parallel_vectorization import ParallelVectorization
from llm_action.src.actions.v38.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v38.implementation.tiling_based_parallelization import TilingBasedParallelization
from llm_action.src.actions.v38.implementation.direct_parallelization import DirectParallelization

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Vectorization,
    ParallelVectorization,
    LoopUnrolling,
    TilingBasedParallelization,
    DirectParallelization,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Vectorization": ["Tiling", "LoopInterchange", "Promotion", "Vectorization", "ParallelVectorization", "LoopUnrolling", "TilingBasedParallelization", "DirectParallelization"],
    "ParallelVectorization": ["Tiling", "LoopInterchange", "Promotion", "Vectorization", "ParallelVectorization", "LoopUnrolling", "TilingBasedParallelization", "DirectParallelization"],
}
