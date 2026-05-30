# Action Enumeration Reasoning — v48 (dataset_matmul)

## Workload Analysis

The input is `linalg.matmul` computing C[I,K] = A[I,J] * B[J,K] in f64. This is a three-deep loop nest with two parallel dimensions (I, K) and one reduction dimension (J). The dataset contains 187 instances with dimensions ranging from 128 to 3072, producing a wide variety of working set sizes.

### Arithmetic Intensity
Matrix multiplication has O(I*J*K) compute on O(I*J + J*K + I*K) data. For large square-ish shapes this gives high arithmetic intensity, meaning the workload is compute-bound when data fits in cache. The critical optimization challenge is ensuring data stays in cache (tiling) and that the ALU is fully utilized (vectorization + parallelization).

### Data Type Implications
f64 means 8 bytes per element. AVX2 provides 256-bit vectors, yielding 4 f64 lanes per vector. FMA (fused multiply-add) enables one multiply and one add per cycle per vector, so peak throughput is 4 FMAs/cycle = 8 FLOPs/cycle per core.

## Target Hardware Considerations

Intel Xeon E5-2680 v4 (Broadwell):
- 28 physical cores (2x14), 2 NUMA nodes, no SMT
- AVX2 + FMA (no AVX-512)
- L1d: 32KB/core (~4K f64 elements), L2: 256KB/core (~32K f64 elements), L3: shared ~tens of MB
- For f64 with AVX2: 4 elements per vector register

### Cache Blocking Targets
- L1-friendly tile: working set of 3 tiles (A-tile, B-tile, C-tile) should fit in ~32KB. For 3 tiles of size TxT: 3*T*T*8 <= 32768 => T ~ 37. Practical: 32x32 tiles.
- L2-friendly tile: 3*T*T*8 <= 262144 => T ~ 104. Practical: 64x64 to 128x128 outer tiles.
- These are rough guides; the RL agent will discover optimal tile sizes through training.

## Intent Rationale

### Intent 1: Cache Locality & Data Reuse (HIGH)

This is the highest-impact optimization family for matrix multiplication. Without tiling, large matrices thrash caches repeatedly — each element of A is reused K times, each element of B is reused I times, but only if they remain in cache across iterations. Tiling partitions the iteration space so that these reuses happen within cache-resident blocks.

**Tiling**: The foundational transformation. Partitions the 3D iteration space into smaller blocks. Multi-level tiling (e.g., L2-level then L1-level) is the standard approach in high-performance BLAS implementations. Tile sizes are the primary tunable parameters.

**Loop Interchange**: The default loop order from `linalg.matmul` may not yield the best spatial locality. For row-major layout, the innermost loop should ideally stride over the fastest-varying dimension of the output (K dimension of C). Interchange reorders the loop nest to achieve this.

### Intent 2: SIMD Compute Throughput (HIGH)

Even with perfect cache behavior, the computation is limited by ALU throughput unless the innermost loops are vectorized. AVX2+FMA can deliver 4 f64 FMAs per cycle, but only if the compiler successfully maps the innermost loop iterations onto vector lanes.

**Vectorization (Sequential Preprocessing)**: Tiles the innermost loop(s) to vector width (4 for f64) using `tile_using_for`, then vectorizes the resulting fixed-extent inner loops. This produces a sequential outer structure suitable for single-threaded or already-parallelized contexts.

**Vectorization (Parallel Preprocessing)**: Tiles the innermost loop(s) to vector width using `tile_using_forall`, which simultaneously distributes the outer tiles across threads. This combines vectorization preparation with parallelism in a single step, potentially enabling better thread-level scheduling.

### Intent 3: Multi-Core Work Distribution (MEDIUM)

The target has 28 physical cores across 2 NUMA nodes. For the larger shapes in the dataset (e.g., 3072x256x128 = ~100M FLOPs), parallel execution across cores provides significant wall-clock reduction. For smaller shapes (128x128x128 = ~4M FLOPs), parallelization overhead may dominate. The RL agent must learn this shape-dependent tradeoff.

**Tiling-based Parallelization**: Tiles one or more outer parallel dimensions and distributes the tiles across threads. This is the more flexible approach — tile size determines granularity, and the tiling can be aligned with cache-blocking tiles for good locality.

**Thread-count Parallelization**: Directly partitions iteration counts by number of threads. Simpler but requires thread count to divide the iteration count evenly. Useful when the parallel dimension is already a convenient multiple of the core count.

The reason both variants are enumerated is that they interact differently with subsequent transformations: tiling-based parallelization creates nested loop structures amenable to further inner tiling/vectorization, while direct thread-count parallelization preserves the original loop structure.
