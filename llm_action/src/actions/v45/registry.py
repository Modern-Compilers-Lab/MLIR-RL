from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v45.implementation.tiling import Tiling
from llm_action.src.actions.v45.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v45.implementation.packing import Packing
from llm_action.src.actions.v45.implementation.promotion import Promotion
from llm_action.src.actions.v45.implementation.vectorization_sequential import VectorizationSequential
from llm_action.src.actions.v45.implementation.vectorization_parallel import VectorizationParallel
from llm_action.src.actions.v45.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v45.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v45.implementation.parallelization_threads import ParallelizationThreads
from llm_action.src.actions.v45.implementation.im2col_lowering import Im2colLowering

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    # Packing,
    Promotion,
    VectorizationSequential,
    VectorizationParallel,
    # LoopUnrolling,
    ParallelizationTiling,
    ParallelizationThreads,
    Im2colLowering,
]

# --- Empirically verified dependency rules (Phase 2, matmul_512_512_1024) ---
# VecSeq/VecPar are TERMINAL: they consume the linalg op and replace it with
# vector ops; no follow-up action can find a tagged linalg op to transform.
# Packing produces a 6D generic that VecSeq's precondition rejects.
ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "VectorizationSequential": [
        "Tiling",
        "LoopInterchange",
        # "Packing",
        "Promotion",
        "VectorizationSequential",
        "VectorizationParallel",
        # "LoopUnrolling",
        "ParallelizationTiling",
        "ParallelizationThreads",
        "Im2colLowering",
    ],
    "VectorizationParallel": [
        "Tiling",
        "LoopInterchange",
        # "Packing",
        "Promotion",
        "VectorizationSequential",
        "VectorizationParallel",
        # "LoopUnrolling",
        "ParallelizationTiling",
        "ParallelizationThreads",
        "Im2colLowering",
    ],
    # "Packing": [
    #     "VectorizationSequential"
    # ],
}

# --- Per-family schedule paths (empirically optimized, Phase 3-4) ---
# Each path is an ordered list of action names representing a valid schedule
# skeleton.  The RL agent walks these paths; "done" is allowed at any path
# terminal.  Paths are ranked by measured performance across 3 representative
# kernels per family on Intel Xeon E5-2680 v4 (Broadwell, AVX2, 28 cores).
#
# Composability constraints baked in:
#   - VecSeq/VecPar always last (terminal actions)
#   - No VecSeq on conv_2d_nchw_fchw or pooling_nchw_max named ops (Post=F)
#   - No Packing on pooling (Pre=F)
#   - Im2col only on conv_2d_nchw_fchw; unlocks VecSeq on resulting generic
#   - Packing → VecSeq blocked (Pre=F on 6D generic)
SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    # matmul: ParTile→Tile→VecSeq is the champion (321x over base, BEATS
    # PyTorch at 1.34x on small kernels). Curated from 11→7 paths; dropped
    # non-parallelized Tiling→Packing/Promotion (weak) and LoopInterchange
    # prefix (never won).
    "matmul": [
        ["ParallelizationTiling", "Tiling", "VectorizationSequential"],
        ["ParallelizationTiling", "VectorizationSequential"],
        ["ParallelizationTiling", "Tiling", "Promotion", "VectorizationSequential"],
        ["ParallelizationTiling", "Tiling", "Promotion"],
        ["ParallelizationThreads", "Tiling", "VectorizationSequential"],
        ["Tiling", "VectorizationSequential"],
    ],
    # conv_2d_nchw_fchw: Im2col→ParTile→Tile→VecSeq champion (40.6x over
    # base). Best 0.25-0.34x of PyTorch (MKL/oneDNN gap). Curated 11→8;
    # dropped Im2col→Tiling→VecSeq (weak without ParTile), standalone
    # Tiling→Promotion (weak), Im2col→ParTile→ParThreads (marginal).
    "conv_2d_nchw_fchw": [
        # Im2col paths (convert to generic, enable VecSeq)
        ["Im2colLowering", "ParallelizationTiling", "Tiling", "VectorizationSequential"],
        ["Im2colLowering", "ParallelizationTiling", "VectorizationSequential"],
        ["Im2colLowering", "ParallelizationTiling", "Tiling", "Promotion", "VectorizationSequential"],
        ["Im2colLowering", "ParallelizationTiling", "Tiling", "Promotion"],
        # Named-op paths (keep conv, use VecPar or Promotion as terminal)
        ["ParallelizationTiling", "ParallelizationThreads", "Tiling", "Promotion"],
        ["ParallelizationTiling", "ParallelizationThreads", "Tiling", "Promotion", "LoopInterchange"],
        ["ParallelizationTiling", "ParallelizationThreads", "Promotion"],
        ["ParallelizationTiling", "ParallelizationThreads", "Promotion", "LoopInterchange"],
        ["ParallelizationTiling", "ParallelizationThreads", "Tiling"],
        ["ParallelizationTiling", "ParallelizationThreads", "Tiling", "LoopInterchange"],
        ["ParallelizationTiling", "Tiling", "VectorizationParallel"],
    ],
    # pooling_nchw: ParTile→Promotion BEATS PyTorch on ALL 3 kernels (up to
    # 2.79x). Curated 9→6; dropped non-parallelized Tiling→VecPar/Promotion
    # (weak without parallelization) and redundant ParThreads→Tiling→VecPar.
    "pooling_nchw": [
        ["ParallelizationThreads"],
        ["ParallelizationTiling", "Promotion"],
        ["ParallelizationTiling", "Tiling", "Promotion"],
        ["ParallelizationTiling", "Tiling"],
        ["ParallelizationTiling", "VectorizationParallel"],
        ["ParallelizationTiling", "ParallelizationThreads", "Tiling", "Promotion"],
    ],
    # add: ParThreads→VecSeq BEATS PyTorch by 1.80x. Memory-bound;
    # parallelization is key. Curated 8→5; dropped VecPar variants
    # (VecSeq consistently better) and non-parallelized Tiling→VecSeq
    # (worse than base on some kernels).
    "add": [
        ["ParallelizationTiling", "Tiling", "VectorizationSequential"],
        ["ParallelizationThreads", "Tiling", "VectorizationSequential"],
        ["ParallelizationThreads", "VectorizationSequential"],
        ["ParallelizationTiling", "VectorizationSequential"],
        ["ParallelizationTiling", "VectorizationParallel"],
    ],
    # relu: ParTile→Tile→VecSeq BEATS PyTorch by 1.25x on medium kernels.
    # ParTile→VecPar best for large. Curated 8→5; dropped ParTile→ParThreads
    # variants (redundant) and non-parallelized Tiling→VecSeq (weak).
    "relu": [
        ["ParallelizationTiling", "Tiling", "VectorizationSequential"],
        ["ParallelizationTiling", "Tiling", "VectorizationParallel"],
        ["ParallelizationTiling", "VectorizationSequential"],
        ["ParallelizationTiling", "VectorizationParallel"],
        ["ParallelizationThreads", "VectorizationSequential"],
    ]
}