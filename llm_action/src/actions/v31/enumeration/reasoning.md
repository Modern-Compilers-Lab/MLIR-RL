# Action Enumeration Reasoning — Pooling NCHW Max (v31)

## Operation Analysis

The benchmark set consists of 250 instances of `linalg.pooling_nchw_max`, a sliding-window max-reduction over NCHW-layout tensors. The loop nest has the following structure:

- **Outer parallel loops**: N (batch), C (channel), OH (output height), OW (output width) — all fully independent.
- **Inner reduction loops**: KH (kernel height), KW (kernel width) — windowed max over the input spatial neighborhood.

### Key Characteristics

1. **Extremely low arithmetic intensity**: Each output element requires loading KH x KW input elements and performing max comparisons. With kernel sizes of 1x1 or 3x3, the compute-to-memory ratio is negligible. This operation is overwhelmingly **memory-bandwidth-bound**.

2. **Large tensor sizes**: Input tensors range up to ~14 GB (e.g., 128x288x224x224xf64). These far exceed any cache level, making efficient data movement the dominant concern.

3. **Strided memory access**: Input elements are accessed with spatial strides (typically stride=2), meaning accesses to the input tensor are not fully contiguous along the innermost dimension.

4. **Tiny reduction windows**: KH and KW are 1 or 3. The inner loops are trivially small, offering minimal compute reuse.

5. **f64 data type**: AVX2 provides 4-wide SIMD for f64 (256-bit vectors).

6. **Embarrassingly parallel outer loops**: N and C dimensions are completely independent, and OH/OW are also independent — ideal for multi-core distribution.

## Optimization Strategy

Given the memory-bound nature, the optimization priority order is:

### 1. Multi-Core Parallelism (HIGH)
With 28 physical cores and large tensor sizes, parallelizing the outer loops (N, C, and/or spatial) is the single most impactful optimization. The batch and channel dimensions are embarrassingly parallel and large enough to distribute evenly.

Two parallelization strategies are relevant:
- **Tiling-based parallelization**: Tile outer dimensions and distribute tiles across threads. More flexible, allows combining with cache tiling.
- **Direct parallelization**: Mark loops as parallel with a thread count. Simpler but less composable with tiling.

### 2. Cache Locality and Data Reuse (HIGH)
Even though arithmetic intensity is low, proper tiling ensures that the input data brought into cache for one output tile is reused across the (small) reduction window before eviction. Tiling the spatial dimensions (OH, OW) so that the corresponding input region fits in L2 (~256KB) or L1 (~32KB) reduces main memory traffic. Loop interchange can improve spatial locality by ensuring the innermost loops traverse contiguous memory.

Transformations:
- **Tiling**: Block the iteration space so working sets fit in cache.
- **Loop Interchange**: Reorder loops to align the innermost iteration with the contiguous memory dimension (W dimension in NCHW layout).

### 3. SIMD Exploitation (MEDIUM)
Vectorizing the max operation across output elements (e.g., along OW) allows 4 f64 max operations per AVX2 instruction. While the operation is memory-bound, vectorization ensures the CPU can consume data at the rate memory delivers it, avoiding becoming a bottleneck. Unrolling the tiny KH/KW reduction loops eliminates loop overhead and exposes more instruction-level parallelism.

Transformations:
- **Vectorization**: Map the innermost parallel loop to SIMD lanes.
- **Unrolling**: Fully unroll the small reduction windows.

## Why Not Other Transformations?

- **Promotion**: Not beneficial here. Arithmetic intensity is too low to justify the overhead of explicitly copying data into temporary buffers. The data is used once (or at most KH x KW = 1-9 times) before moving on.
- **Packing/Layout Transformation**: The NCHW layout already provides reasonable spatial locality for the access pattern. Layout changes would add overhead without sufficient reuse to amortize.
- **Im2col/Lowering**: Not applicable — this is a pooling operation, not a convolution that can be lowered to a contraction.
- **Fusion**: Pooling is a standalone operation in this benchmark; there are no adjacent producers/consumers to fuse with.
