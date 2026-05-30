from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v50.implementation.tiling import Tiling
from llm_action.src.actions.v50.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v50.implementation.promotion import Promotion
from llm_action.src.actions.v50.implementation.sequential_vectorization import SequentialVectorization
from llm_action.src.actions.v50.implementation.parallel_vectorization import ParallelVectorization
from llm_action.src.actions.v50.implementation.tiling_parallelization import TilingParallelization
from llm_action.src.actions.v50.implementation.thread_parallelization import ThreadParallelization

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    # Promotion,
    SequentialVectorization,
    ParallelVectorization,
    TilingParallelization,
    ThreadParallelization,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    # SeqVec moves tag from linalg op to outermost scf.for, making subsequent
    # actions unable to find the tagged operation (all 7 post=F after SeqVec).
    "SequentialVectorization": [
        "Tiling",
        "LoopInterchange",
        # "Promotion",
        "ParallelVectorization",
        "TilingParallelization",
        "ThreadParallelization",
    ],
}

SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    "pooling_nchw": [
        # Winner for small (1.04x) and large (3.10x) kernels. Canonical parallelize-tile-vectorize.
        ["TilingParallelization", "Tiling", "SequentialVectorization"],
        # Winner for medium kernel (3.40x). Thread-based parallelism suits larger batch+channel dims.
        ["ThreadParallelization", "Tiling", "SequentialVectorization"],
        # Competitive on large kernel (3.03x). Hierarchical tiling for deeper cache blocking.
        ["TilingParallelization", "Tiling", "Tiling", "SequentialVectorization"],
        # 3rd best on medium (3.21x). Interchange entry enables different loop ordering.
        ["LoopInterchange", "TilingParallelization", "SequentialVectorization"],
        # Competitive across all subsets (1.01-3.03x). Different mechanism (generic + forall).
        ["ParallelVectorization", "Tiling", "SequentialVectorization"],
    ],
}
