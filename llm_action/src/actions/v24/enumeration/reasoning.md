# Layer-1 Action Enumeration Reasoning — v24

## Input Analysis

The input templates describe `linalg.matmul` operations — classic 3-loop-nest computations with two parallel dimensions (I, K) and one reduction dimension (J) operating on f64 tensors. The concrete instance is a 128x256 by 256x128 matmul producing a 128x128 result.

## Hardware Context

Target: Intel Xeon E5-2680 v4 (Broadwell)
- 28 physical cores (2 sockets x 14 cores), 2 NUMA nodes, no SMT
- AVX2 + FMA (256-bit vectors → 4 FP64 lanes per register)
- Cache hierarchy: L1d ~32KB/core, L2 ~256KB/core, L3 ~35MB/socket
- No AVX-512

## Performance Analysis

For this workload class on this hardware, performance is dominated by three factors:

1. **Data movement**: Tensor operands (e.g., 128x256xf64 = 256KB for one operand) can exceed L1/L2 capacity. Without tiling and data relocation, the kernel becomes memory-bound due to cache capacity and conflict misses.

2. **Instruction throughput**: Without SIMD vectorization, only 1 of 4 available FP64 lanes is utilized per cycle. The FMA unit can perform a multiply-add per cycle per lane, so vectorization provides up to 4x throughput gain.

3. **Core utilization**: Without thread-level parallelism, 27 of 28 cores are idle. For matrices of sufficient size, distributing the outer parallel loops provides near-linear scaling.

## Intent Organization

I organize the 9 candidate transformations into 3 intents, each addressing a distinct performance axis:

### Intent 1: Cache-Aware Data Movement (HIGH)

Addresses how data flows through the memory hierarchy. Contains:
- **Tiling**: Controls iteration space blocking to create cache-fitting working sets.
- **Promotion**: Materializes tiled operand slices into contiguous local buffers, eliminating conflict misses. Includes internal bufferization.
- **Packing**: Reorganizes source data layout to match tiled access order, converting strided accesses to contiguous.

Rationale for HIGH: Cache blocking is consistently the single most impactful optimization for dense loop nests on CPUs with deep cache hierarchies. A well-tiled matmul can achieve 10-50x speedup over naive execution.

### Intent 2: Loop Nest Structure Optimization (HIGH)

Restructures the iteration space without changing data layout. Contains:
- **Loop Interchange**: Reorders loops for stride-1 innermost access, critical for cache line utilization and vectorization alignment.
- **Loop Unrolling**: Exposes ILP by replicating the loop body, filling superscalar pipeline slots and amortizing loop overhead.
- **Loop Peeling**: Separates remainder iterations so the main loop operates on clean multiples of tile/vector widths.

Rationale for HIGH: Loop ordering directly determines memory stride patterns (and thus cache behavior), and unrolling is essential for keeping the FMA pipeline fed. These transformations are prerequisites for effective vectorization.

### Intent 3: Parallelism Exposure (HIGH)

Exposes parallelism at multiple hardware levels. Contains:
- **Vectorization**: Maps computations to AVX2 SIMD lanes (4 FP64 elements per vector operation).
- **Parallelization**: Distributes parallel loop iterations across the 28 physical cores.
- **Canonicalization**: Normalizes IR between transformation steps, folding redundant operations and cleaning up artifacts so that vectorization and parallelization passes can pattern-match cleanly.

Rationale for HIGH: Vectorization alone provides a 4x throughput multiplier for FP64 and is non-negotiable for compute-bound kernels. Thread parallelism adds further scaling proportional to core count. Canonicalization is the enabling glue that keeps the IR in a form amenable to these passes.

## Design Decisions

- **No kernel-specific actions**: All transformations are framed in generic loop-nest terms. "Tiling" is not "Tile the M dimension" — dimension targeting is deferred to Layer 2.
- **No compound actions**: Each transformation is a single macro RL action. "Tile and Fuse" is split. Promotion internally includes bufferization because they are inseparable (promotion requires memref form).
- **Canonicalization as an explicit action**: Rather than implicitly running canonicalization, making it an explicit RL action gives the agent control over when to normalize IR. This is important because premature canonicalization can destroy structure needed by other passes, while delayed canonicalization can leave the IR in a state that blocks downstream transformations.
- **3 intents, 9 transformations**: This provides a balanced, non-overlapping coverage of the optimization space. Each intent maps to a distinct hardware resource (cache hierarchy, execution pipeline, parallel units).
