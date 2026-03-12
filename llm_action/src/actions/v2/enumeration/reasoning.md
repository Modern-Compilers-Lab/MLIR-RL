# Layer 1 — Action Enumeration Reasoning (v2)

## Input Analysis

The RL system operates on three classes of structured numerical kernels expressed as MLIR linalg operations:

1. **Matrix Multiplication** (`linalg.matmul`): A rank-2 contraction with iteration space (I, J, K) where I and K are parallel and J is a reduction. Example shape: 256×512 @ 512×1024. This is a classic compute-bound loop nest with O(N³) arithmetic on O(N²) data — performance is dominated by data reuse in the cache hierarchy.

2. **2D Convolution** (`linalg.conv_2d_nchw_fchw`): A deep loop nest with 7 loops (N, F, C, OH, OW, KH, KW). Parallel loops: N, F, OH, OW. Reduction loops: C, KH, KW. The iteration space is large and has complex memory access patterns due to sliding-window semantics. Data reuse patterns differ across loops (filter reuse vs. input feature map reuse).

3. **Generic Elementwise** (`linalg.generic` with all-parallel iterators): A 5D pointwise operation with no reductions. Memory-bandwidth bound since arithmetic intensity is low (one add per load-store pair). Performance depends on memory throughput and vectorization efficiency.

## Hardware Context

- **Intel Xeon E5-2680 v4 (Broadwell)**: AVX2 + FMA, 256-bit vectors (4 FP64 lanes), no AVX-512.
- **Cache**: L1d 32KB, L2 256KB, L3 ~35MB shared per socket.
- **Cores**: 28 physical cores across 2 NUMA nodes, no SMT.

## Optimization Intent Reasoning

### Intent 1: Data Locality Optimization (HIGH Priority)

For all three kernel types, improving cache utilization is critical:
- **Matmul** has high arithmetic intensity but only if data tiles fit in L1/L2. Without tiling, streaming through large matrices causes constant cache misses.
- **Convolution** has even more complex reuse patterns. Tiling across output spatial and channel dimensions can keep filter tiles and input patches in cache.
- **Elementwise** ops are memory-bound; tiling helps with prefetch friendliness and keeps working sets in cache even though arithmetic intensity is low.

Tiling (blocking) is the primary mechanism. Loop interchange complements tiling by reordering loops to maximize stride-1 accesses and improve spatial locality within tiles. Together, these form the foundation of any high-performance loop nest schedule.

### Intent 2: SIMD and Compute Throughput (HIGH Priority)

The target CPU has AVX2+FMA capable of 4 FP64 FMA operations per cycle per core. Exploiting this requires:
- **Vectorization** of innermost loops along contiguous memory dimensions to utilize 256-bit vector registers.
- **Unrolling** to expose instruction-level parallelism, fill the FMA pipeline, hide latency, and reduce loop overhead.

For matmul and convolution, vectorization typically targets the innermost parallel dimension of the output. For elementwise, any dimension with contiguous memory access is suitable. Unrolling amplifies the benefit by keeping the FMA units fed.

### Intent 3: Coarse-Grain Parallelism (MEDIUM Priority)

With 28 physical cores, coarse-grain parallelism across outer loop dimensions is important for large problems:
- Matmul and convolution have multiple parallel outer loops (batch, output channels, spatial dimensions) that can be distributed across threads.
- Elementwise operations are embarrassingly parallel.

However, parallelization must avoid oversubscription and be NUMA-aware. It is medium priority because single-core performance (tiling + vectorization) must be addressed first — parallelization of a poorly-tiled kernel just multiplies cache misses. The RL agent should learn to apply parallelization after establishing good single-core schedules.

## Transformation Selection

Under these three intents, I enumerate the following macro RL transformations:

**Data Locality:**
- **Tiling**: The most impactful single transformation for loop nests. Partitions iteration spaces into blocks that fit in cache levels.
- **Loop Interchange**: Reorders loops to improve spatial locality (stride-1 access patterns) and enable better tiling configurations.
- **Packing**: Copies data into contiguous buffers with favorable layout for the tiled computation, eliminating TLB misses and conflict misses from non-unit strides.

**SIMD/Compute:**
- **Vectorization**: Maps innermost loops onto SIMD instructions (AVX2, 4 FP64 lanes). Essential for utilizing compute throughput.
- **Loop Unrolling**: Replicates loop bodies to reduce branch overhead, expose ILP, and enable register-level reuse. Complements vectorization by keeping vector pipelines full.

**Parallelism:**
- **Parallelization**: Distributes outer parallel loops across CPU cores using OpenMP-style work partitioning.
- **Fusion**: Merges producer-consumer loop nests to reduce intermediate materialization, improve locality, and reduce synchronization barriers in parallel contexts.

## Deduplication and Granularity Check

All 7 transformations are distinct macro actions:
- No transformation is a dimension-specific variant of another.
- Each represents a different compiler optimization category with independent parameters.
- Each can be expressed as a single RL action with parameters determined by Layer 2.

The selection covers the critical optimization categories for CPU loop-nest performance: memory hierarchy (tiling, interchange, packing), compute throughput (vectorization, unrolling), and parallelism (parallelization, fusion).
