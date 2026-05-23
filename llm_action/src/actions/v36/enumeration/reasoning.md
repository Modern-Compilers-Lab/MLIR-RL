# Action Enumeration Reasoning — v36 (dataset_conv2d_img2col)

## Workload Analysis

The benchmark set consists of 278 instances of im2col-lowered 2D convolution (NCHW/FCHW layout). The im2col transformation has already been applied at the IR level: each instance contains:

1. **Im2col extraction** — a parallel 3D `linalg.generic` that gathers image patches into a contiguous column buffer of shape `[N, C*KH*KW, OH*OW]`.
2. **Contraction (tagged `operation_0`)** — a 4D `linalg.generic` performing a batched matrix multiplication: `filter[F, C*KH*KW] × img2col[N, C*KH*KW, OH*OW] → output[N, F, OH*OW]`, with iterator types `[parallel, parallel, parallel, reduction]`.

Per system notes, the RL agent targets the **contraction operation** (`operation_0`), which is the dominant compute kernel. The surrounding tensor reshapes (collapse/expand) and the im2col extraction are structural and not primary optimization targets.

## Shape Characteristics

Across the dataset:
- **Batch (d0 / N):** 128 or 256 — large, providing ample outer parallelism.
- **Filters (d1 / F):** 48–512 — varies widely, affecting tile size choices.
- **Output spatial (d2 / OH*OW):** 9–225 — moderate variation; this is the "N" dimension of the batched matmul equivalent.
- **Reduction (d3 / C*KH*KW):** 48–2592 — depends on channel count and kernel size. For 1×1 kernels this equals C; for 3×3 or 7×7 it grows by KH*KW.

The contraction is a **batched GEMM** pattern: for each batch element, it computes `filter[F, K] @ img2col[K, S] → output[F, S]` where K = C*KH*KW and S = OH*OW. Classical matmul optimization strategies apply directly.

## Target Hardware Considerations

- **Intel Xeon E5-2680 v4**: 28 physical cores (2 sockets × 14), AVX2+FMA, no AVX-512.
- **FP64**: 4 vector lanes per 256-bit register; FMA throughput is the performance ceiling.
- **Cache**: L1d 32KB, L2 256KB per core, shared L3 ~35MB per socket. Tiling to L1/L2 is critical for reuse.
- **Parallelism**: 28 cores available; batch dimension (128/256) divides cleanly for thread distribution.

## Optimization Intent Design

### Intent 1: Cache Locality and Data Reuse (HIGH)

The contraction has three parallel dimensions and one reduction dimension. The filter matrix (`[F, C*KH*KW]`) is reused across all batch elements and spatial positions; the img2col buffer (`[N, C*KH*KW, OH*OW]`) is reused across all filters. Without tiling, working sets for moderate-to-large shapes far exceed L2 capacity (e.g., for N=128, F=192, K=128, S=49: filter alone is 192×128×8B ≈ 192KB, and a single batch slice of img2col is 128×49×8B ≈ 50KB).

Tiling partitions the iteration space so that the active working set fits in L1 or L2. Loop interchange ensures the innermost loop has stride-one memory access. Promotion copies tiled operand slices into contiguous temporary buffers, removing non-unit-stride penalties and enabling efficient downstream vectorization.

Transformations selected:
1. **Tiling** — the foundational transformation; creates manageable blocks.
2. **Loop Interchange** — reorders loops for contiguous access patterns.
3. **Promotion** — ensures contiguous layout of tiled operands after bufferization.

### Intent 2: SIMD Exploitation and Compute Throughput (HIGH)

FP64 on AVX2 gives 4-wide SIMD lanes and FMA capability. Without vectorization, the core operates at 1/4 of peak FP64 throughput. The contraction's multiply-accumulate body (`mulf` + `addf`) maps directly to FMA vector instructions. After tiling, the innermost tile dimensions have fixed, known extents that can be matched to the vector width.

Unrolling inner loops exposes multiple independent FMA operations per cycle, keeping the out-of-order execution pipeline saturated and reducing loop-control overhead relative to useful computation.

Transformations selected:
1. **Vectorization** — maps inner loops to AVX2 SIMD operations.
2. **Unrolling** — exposes ILP to fill the FMA pipeline.

### Intent 3: Multi-Core Work Distribution (MEDIUM)

The target machine has 28 physical cores. The contraction has three parallel dimensions with large trip counts (batch up to 256, filters up to 512, spatial up to 225). Without parallelization, only one core is utilized.

The batch dimension is the most natural candidate for parallelization (large, independent, NUMA-friendly). Two parallelization strategies are provided: tiling-based (more flexible, works with any trip count, creates coarse work chunks) and direct num_threads-based (simpler, requires divisibility).

This intent is ranked MEDIUM rather than HIGH because the primary performance bottleneck for a single core is cache and SIMD efficiency — parallelization scales the optimized single-core performance but does not improve it. Additionally, cache locality transformations (tiling, promotion) should be established first before parallelization to avoid scaling cache-miss-dominated code.

Transformations selected:
1. **Parallelization (Tiling-Based)** — creates coarse parallel tiles distributed to threads.
2. **Parallelization (Direct)** — distributes iterations directly to a fixed thread count.
