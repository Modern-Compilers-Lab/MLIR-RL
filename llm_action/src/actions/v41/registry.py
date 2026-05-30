from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v41.implementation.tiling import Tiling
from llm_action.src.actions.v41.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v41.implementation.promotion import Promotion
from llm_action.src.actions.v41.implementation.vectorization_seq import VectorizationSeq
from llm_action.src.actions.v41.implementation.vectorization_par import VectorizationPar
from llm_action.src.actions.v41.implementation.unrolling import Unrolling
from llm_action.src.actions.v41.implementation.parallelization_tile import ParallelizationTile
from llm_action.src.actions.v41.implementation.parallelization_threads import ParallelizationThreads
from llm_action.src.actions.v41.implementation.packing import Packing

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    # LoopInterchange,
    # Promotion,
    VectorizationSeq,
    VectorizationPar,
    # Unrolling,
    ParallelizationTile,
    ParallelizationThreads,
    # Packing,
]

# Empirical composability denylist (matmul_512_512_512, v41 actions).
# Schema: "<BlockerActionName>": ["<BlockedActionName>", ...]
# Semantics: if action X (key) has executed in the episode, actions Y (values)
# become unavailable.  Verified by direct tool calls — only truly structural
# failures are listed.  Parameter-dependent failures are NOT listed since RL
# tunes parameters to fit.  Self-edges omitted (handled by unique_execution).
ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "VectorizationSeq": [  # terminal: consumes linalg op, blocks everything
        "Tiling",
        # "LoopInterchange",
        # "Promotion",
        "VectorizationPar",
        # "Unrolling",
        "ParallelizationTile",
        "ParallelizationThreads",
        # "Packing",
    ],
    "VectorizationPar": [  # terminal: consumes linalg op, blocks everything
        "Tiling",
        # "LoopInterchange",
        # "Promotion",
        "VectorizationSeq",
        # "Unrolling",
        "ParallelizationTile",
        "ParallelizationThreads",
        # "Packing",
    ],
    # "Packing": [  # 6D generic — interchange is no-op, vectorization can't handle it
    #     # "LoopInterchange",
    #     "VectorizationSeq",
    #     "VectorizationPar",
    # ],
    # "Unrolling": [  # unrolled copies can't be packed
    #     "Packing",
    # ],
}

# Per-family allowlist of high-value schedule shapes (action sequences).
# Each entry is a list of action class names in application order.
# These were selected from Phases 3-4 exploration: they compose, execute
# successfully, and achieve >50x speedup on at least one kernel size.
# Parameters (tile sizes, vector sizes, thread counts) are tuned by RL.
SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    "matmul": [
        # 1-step: VecPar alone — best for small/medium (0.18ms small, 2.2ms medium)
        ["VectorizationPar"],
        # 2-step: ParTile+VecPar — strong 2-step (2.4ms medium, 3.3ms large)
        ["ParallelizationTile", "VectorizationPar"],
        # 2-step: ParTile+VecSeq — strong 2-step (2.6ms medium, 3.8ms large)
        ["ParallelizationTile", "VectorizationSeq"],
        # 3-step: ParTile+Tile+VecSeq — best for large (2.0ms), robust across sizes
        ["ParallelizationTile", "Tiling", "VectorizationSeq"],
        # 2-step: ParTile+Promotion — good all-around (2.5ms medium, 3.5ms large)
        # ["ParallelizationTile", "Promotion"],
        # 3-step: ParThreads+Tile+VecSeq — strong alternative (3.1ms medium)
        ["ParallelizationThreads", "Tiling", "VectorizationSeq"],
        # 2-step: ParThreads+VecPar — compact and effective (3.2ms medium)
        ["ParallelizationThreads", "VectorizationPar"],
        # 3-step: ParTile+Tile+Promotion — decent for small/medium, avoid for large
        # ["ParallelizationTile", "Tiling", "Promotion"],
    ],
}
