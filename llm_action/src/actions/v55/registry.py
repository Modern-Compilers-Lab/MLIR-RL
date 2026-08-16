from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v55.implementation.tiling import Tiling
from llm_action.src.actions.v55.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v55.implementation.promotion import Promotion
from llm_action.src.actions.v55.implementation.vectorization_seq import VectorizationSeq
from llm_action.src.actions.v55.implementation.vectorization_par import VectorizationPar
from llm_action.src.actions.v55.implementation.unrolling import Unrolling
from llm_action.src.actions.v55.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v55.implementation.parallelization_threads import ParallelizationThreads
from llm_action.src.actions.v55.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v55.implementation.packing import Packing

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    VectorizationSeq,
    VectorizationPar,
    Unrolling,
    ParallelizationTiling,
    ParallelizationThreads,
    Im2colLowering,
    Packing,
]

# Cross-kernel denylist: actions whose precondition ALWAYS fails for a family,
# or whose postcondition ALWAYS fails regardless of IR state.
# Key: kernel family name. Value: list of action class names that are never valid.
# Empirically validated on representative kernels per family (v55 MCP tools).
ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "matmul": ["Im2colLowering"],
    "conv2d": ["VectorizationSeq", "VectorizationPar"],
    "pooling": ["Im2colLowering", "VectorizationSeq", "VectorizationPar"],
    "add": ["Im2colLowering"],
    "relu": ["Im2colLowering"],
}

# Per-family allowlist of empirically validated schedule paths (action-sequence skeletons).
# Each path is a list of action class names applied in order.
# Vectorization actions (VectorizationSeq, VectorizationPar) are TERMINAL:
# no further actions can compose after them (tag moves from linalg op to scf loop).
# Pruned to top 5-8 per family by average speedup over base MLIR across representative
# kernels per family. Benchmarked on Intel Xeon E5-2680 v4 (28 cores, AVX2).
SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    "matmul": [
        # ~276x med — par tiles + seq vectorization (best single-kernel)
        ["ParallelizationTiling", "VectorizationSeq"],
        # ~271x avg — 3-level: par tiles, cache tiles, seq vec (best cross-kernel)
        ["ParallelizationTiling", "Tiling", "VectorizationSeq"],
        # ~193x avg — par tiles + par vectorization
        ["ParallelizationTiling", "VectorizationPar"],
        # ~186x avg — single-step vectorization, nearly as good
        ["VectorizationPar"],
        # ~155x avg — loop reorder improves vectorization access pattern
        ["LoopInterchange", "VectorizationPar"],
        # ~193x avg — 3-level cache tiling between par tiles and par vec
        ["ParallelizationTiling", "Tiling", "VectorizationPar"],
        # ~55x avg — two-level tiling for cache locality (non-terminal)
        ["ParallelizationTiling", "Tiling"],
        # ~42x — packing restructures data layout, then parallelism (v55 NEW)
        ["Packing", "ParallelizationTiling"],
        # ~25x avg — simple parallelization baseline (non-terminal)
        ["ParallelizationTiling"],
    ],
    "conv2d": [
        # ~32x avg — thread par + cache tiling (best cross-kernel, agent-validated)
        ["ParallelizationThreads", "Tiling"],
        # ~27x avg — parallel tiles + reduction tiling for cache locality
        ["ParallelizationTiling", "Tiling"],
        # ~28x avg — loop reorder + parallel tiles + tiling
        ["LoopInterchange", "ParallelizationTiling", "Tiling"],
        # ~28x avg — loop reorder + parallel tiles
        ["LoopInterchange", "ParallelizationTiling"],
        # ~25x avg — single-step thread parallelism
        ["ParallelizationThreads"],
        # ~23x avg — single-step parallel tiling
        ["ParallelizationTiling"],
        # ~14x avg — im2col converts conv to matmul-like, then parallelize
        ["Im2colLowering", "ParallelizationTiling"],
        # ~8x avg — promotion + thread parallelism
        ["Promotion", "ParallelizationThreads"],
    ],
    "pooling": [
        # ~10x, 1.3-4.7x vs torch — beats PyTorch on ALL sizes (agent-validated)
        ["ParallelizationThreads"],
        # ~6.6x, ~3.0x vs torch — multi-dim parallel tiling
        ["ParallelizationTiling"],
        # Note: Compositions (TP→ThP, ThP→T, ThP→TP) all degrade performance
        # vs single ThreadPar. Tiling/Promotion/Packing are harmful (0.06-0.46x).
    ],
    "add": [
        # ~6.5x, ~1.3x vs torch — beats PyTorch on medium/large
        ["ParallelizationThreads"],
        # ~6.4x, ~1.3x vs torch — multi-dim parallel tiling, beats PyTorch
        ["ParallelizationTiling"],
        # ~5.9x — two-level parallelism (tiles then threads)
        ["ParallelizationTiling", "ParallelizationThreads"],
        # ~2.8x avg — vectorization with implicit parallelism
        ["VectorizationPar"],
        # ~2.5x avg — parallel tiles then sequential vectorization
        ["ParallelizationTiling", "VectorizationSeq"],
        # ~1.6x avg — parallel tiles then parallel vectorization
        ["ParallelizationTiling", "VectorizationPar"],
    ],
    "relu": [
        # ~6.5x, 1.1-1.9x vs torch — beats PyTorch on ALL sizes (agent-validated)
        ["ParallelizationThreads"],
        # ~5.4x — multi-dim parallel tiling (agent-validated: 12.16ms)
        ["ParallelizationTiling"],
        # Note: Vectorization is HARMFUL for relu (0.36-0.72x alone, degrades
        # parallelization when composed). Not included as schedule paths.
    ],
}
