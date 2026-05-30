# Layer 1 — Action Enumeration Reasoning (v44)

## Benchmark Set Analysis

The `dataset_ml` training set comprises 1135 instances across five operation families:

| Family | Count | Loop Structure | Compute Profile |
|--------|-------|---------------|-----------------|
| conv_2d_nchw_fchw | 277 | 7 loops (4 parallel + 3 reduction) | Compute-bound |
| matmul | 186 | 3 loops (2 parallel + 1 reduction) | Compute-bound |
| pooling_nchw_max | 249 | 6 loops (4 parallel + 2 reduction) | Mixed |
| add | 270 | 4 parallel loops | Memory-bound |
| relu | 148 | 2–4 parallel loops | Memory-bound |

### Loop-Nest Perspective

From a generic loop-nest viewpoint, all five families share common optimization opportunities:

1. **Iteration space blocking (tiling)** — Every kernel has multi-dimensional loop nests where working sets can exceed cache capacity. Tiling is universally applicable.

2. **Loop ordering** — NCHW data layout means the innermost storage dimension (W) should ideally correspond to the innermost loop for stride-1 access. Non-obvious orderings arise after tiling or when reduction loops are interleaved with parallel loops.

3. **Vectorization** — All kernels have at least one dimension that can be mapped to SIMD lanes. FP64 on AVX2 gives 4 lanes per 256-bit register. The innermost loop must be tiled to vector width for efficient lowering.

4. **Parallelism** — With 28 physical cores (2 NUMA nodes), outer parallel loop dimensions must be distributed across threads. The compute-bound kernels (matmul, conv) benefit from near-linear scaling; memory-bound kernels (add, relu) benefit from aggregate bandwidth.

5. **Data layout restructuring** — The conv_2d family (24% of the dataset) has irregular windowed access patterns in the reduction loops. Lowering to a matmul-like contraction (im2col) regularizes these patterns for all downstream transforms.

6. **Data packing/promotion** — After tiling, operand slices may have non-unit stride. Packing reorganizes data into contiguous blocks; promotion copies tiled slices into local buffers for guaranteed contiguous reuse.

## Hardware Constraints Informing Priority

- **L1 32KB, L2 256KB**: Tile sizes for compute-bound kernels must fit working sets in L1 (micro-tiles) and L2 (macro-tiles). This makes tiling the highest-impact transformation.
- **AVX2 FP64 = 4 lanes**: Vectorization width is constrained to 4 for double precision. Sequential vectorization is the baseline; parallel vectorization combines SIMD with thread distribution.
- **28 cores, no SMT**: Coarse-grain parallelism is essential. Over-decomposition (too many small tiles) wastes overhead; under-decomposition leaves cores idle.
- **FMA units**: Loop unrolling exposes independent FMA operations for pipelining, critical for compute-bound kernels at 2 FMAs/cycle/core.
- **NUMA (2 sockets)**: Large tensors (common in the dataset, e.g., 128×128×112×112 for pooling) span NUMA boundaries. Thread-tile mapping should respect socket locality.

## Intent Design Rationale

### Intent 1: Cache Locality and Data Reuse (HIGH)

This intent addresses the most fundamental bottleneck: data movement. For compute-bound kernels (matmul, conv — 41% of dataset), tiling creates temporal data reuse that converts O(N³)-class memory traffic to O(N²). For memory-bound kernels (add, relu, pooling — 59%), tiling improves spatial locality, TLB utilization, and prefetcher effectiveness. Loop interchange ensures stride-1 access on the innermost loops. Packing and promotion further guarantee contiguous access within tiles.

Four transformations are included:
- **Tiling**: The foundation — partitions iteration space into cache-fitting blocks.
- **Loop Interchange**: Ensures optimal loop ordering for memory access patterns.
- **Packing**: Reorganizes data into contiguous tile-aligned layouts.
- **Promotion**: Copies tiled operand slices into contiguous local buffers (requires internal bufferization).

### Intent 2: SIMD and Instruction-Level Parallelism (HIGH)

Vectorization is the primary mechanism to exploit AVX2 on this hardware. Every kernel in the dataset has at least one dimension amenable to SIMD. The two vectorization variants (sequential vs. parallel preprocessing) correspond to distinct RL action choices: sequential vectorization tiles inner loops with for-loops (no threading overhead, simpler), while parallel vectorization uses forall-loops to simultaneously distribute outer tiles to threads. Loop unrolling complements vectorization by exposing independent operations for FMA pipelining and reducing loop control overhead.

Three transformations:
- **Vectorization (Sequential)**: SIMD lowering with sequential outer tiling.
- **Vectorization (Parallel)**: SIMD lowering with parallel outer distribution.
- **Loop Unrolling**: Reduces loop overhead and exposes ILP for out-of-order execution.

### Intent 3: Thread-Level Parallelism and Iteration Space Restructuring (HIGH)

The 28-core target demands effective thread-level parallelism. Parallelization via tiling gives the RL agent control over tile granularity (matching cache and NUMA boundaries), while direct thread mapping provides simpler scheduling. Im2col lowering is included here because it fundamentally restructures the convolution iteration space into a regular matrix contraction, which is a prerequisite for effective parallelization (and all other optimizations) on the resulting dense loop nest. This benefits 277 conv instances (24% of dataset). Note that im2col produces a contraction as the primary compute operation; subsequent optimizations should target this contraction.

Three transformations:
- **Parallelization (Tiling)**: Tile and distribute outer parallel dimensions to threads.
- **Parallelization (Threads)**: Directly map parallel iterations to a thread pool.
- **Im2col Lowering**: Convert convolution to matmul-like contraction for regular parallel work.

## Transformation Count Summary

| Intent | Priority | Transformations |
|--------|----------|-----------------|
| Cache Locality and Data Reuse | HIGH | 4 (Tiling, Loop Interchange, Packing, Promotion) |
| SIMD and Instruction-Level Parallelism | HIGH | 3 (Vectorization Sequential, Vectorization Parallel, Loop Unrolling) |
| Thread-Level Parallelism and Iteration Space Restructuring | HIGH | 3 (Parallelization Tiling, Parallelization Threads, Im2col Lowering) |
| **Total** | | **10 macro RL actions** |

All three intents are rated HIGH because the benchmark set spans both compute-bound and memory-bound kernels at significant tensor sizes, and the target hardware (multi-core, wide SIMD, deep cache hierarchy) demands all three optimization axes for competitive performance.
