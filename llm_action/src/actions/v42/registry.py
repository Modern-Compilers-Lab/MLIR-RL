from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v42.implementation.tiling import Tiling
from llm_action.src.actions.v42.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v42.implementation.promotion import Promotion
from llm_action.src.actions.v42.implementation.packing import Packing
from llm_action.src.actions.v42.implementation.vectorization_sequential import VectorizationSequential
from llm_action.src.actions.v42.implementation.vectorization_parallel import VectorizationParallel
from llm_action.src.actions.v42.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v42.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v42.implementation.parallelization_threads import ParallelizationThreads

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Packing,
    VectorizationSequential,
    VectorizationParallel,
    LoopUnrolling,
    ParallelizationTiling,
    ParallelizationThreads,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    # VecSeq is terminal: tag migrates from linalg op to scf.for loop,
    # all 8 other actions fail postcondition after VecSeq (Phase 2, 8/8 blocked)
    "VectorizationSequential": [
        "Tiling",
        "LoopInterchange",
        "Promotion",
        "Packing",
        "VectorizationParallel",
        "LoopUnrolling",
        "ParallelizationTiling",
        "ParallelizationThreads",
    ],
    # VecPar is terminal: tag migrates from linalg op to scf.forall loop,
    # all 8 other actions fail postcondition after VecPar (Phase 2, 8/8 blocked)
    "VectorizationParallel": [
        "Tiling",
        "LoopInterchange",
        "Promotion",
        "Packing",
        "VectorizationSequential",
        "LoopUnrolling",
        "ParallelizationTiling",
        "ParallelizationThreads",
    ],
    # LoopUnrolling creates multiple matmul copies; Packing postcondition
    # cannot verify the result (Phase 2: Pre=T, Post=F)
    "LoopUnrolling": [
        "Packing"
    ],
}

SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    "matmul": [
        # Standalone VecPar: universal winner across all matmul sizes
        # Best: 187.8x on 512x128, 154.7x on 512x256, 22.0x on 128x128
        ["VectorizationParallel"],
        # Standalone VecSeq: sequential vectorization alternative
        # Best: 7.57x on 512x128; useful when parallelism is constrained
        ["VectorizationSequential"],
        # ParTile + VecPar: explicit work distribution + vectorization
        # Best: 118.8x on 2048x512, viable fallback with nested scf.forall
        ["ParallelizationTiling", "VectorizationParallel"],
        # Tiling + VecPar: cache-tiling + vectorization (classic HPC pattern)
        # Probe: 1.06x on 512x256; structurally sound, RL tunes tile sizes
        ["Tiling", "VectorizationParallel"],
        # ParTile + Tiling + VecPar: canonical HPC (distribute → tile → vectorize)
        # Best 3-step: 132.6x on 512x256; gap vs VecPar narrows at large sizes (1.10x on 2048x512)
        ["ParallelizationTiling", "Tiling", "VectorizationParallel"],
        # --- VecSeq mirrors (parallel distribution via ParTile + sequential vectorization) ---
        # ParTile + VecSeq: parallel distribution + sequential vectorization
        # Probe: 89.8x on 512x256; ParTile provides scf.forall parallelism
        ["ParallelizationTiling", "VectorizationSequential"],
        # Tiling + VecSeq: cache-tiling + sequential vectorization
        # Probe: 11.3x on 512x256; sequential-only, no parallelism
        ["Tiling", "VectorizationSequential"],
        # ParTile + Tiling + VecSeq: canonical HPC with sequential vectorization
        # Probe: 141.6x on 512x256; BEATS VecPar equivalent (0.59ms vs 0.63ms)
        ["ParallelizationTiling", "Tiling", "VectorizationSequential"],
    ],
}