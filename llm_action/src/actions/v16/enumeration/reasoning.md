# Layer 1 — Action Enumeration Reasoning (v16)

## Input Analysis

The RL training input is a single `linalg.conv_2d_nchw_fchw` operation — a 2D convolution with NCHW input layout and FCHW filter layout, stride 1, dilation 1. This is a canonical dense loop-nest kernel with 7 nested loops:

- **Parallel loops**: N (batch), F (output channels), OH (output height), OW (output width)
- **Reduction loops**: C (input channels), KH (kernel height), KW (kernel width)

From a loop-nest perspective, this is a 7-deep nested loop with 4 parallel and 3 reduction dimensions, exhibiting significant data reuse across multiple dimensions. The operation is compute-bound for large problem sizes and memory-bound for small ones.

## Target Hardware Context

Intel Xeon E5-2680 v4 (Broadwell):
- 28 physical cores, 2 NUMA nodes, no HT
- AVX2 + FMA (256-bit vectors: 4 FP64, 8 FP32 lanes)
- Cache: L1d 32KB, L2 256KB, L3 ~35MB shared per socket
- No AVX-512

## Optimization Intent Reasoning

### Intent 1: Data Locality and Cache Optimization (HIGH)

This is the highest-impact optimization category for this workload. The 7-deep loop nest accesses three large tensors (input, filter, output) with complex reuse patterns. Without cache-aware blocking, working sets will far exceed L1/L2 capacity, causing frequent cache misses that dominate execution time.

**Tiling** is the foundational transformation — it partitions the iteration space into blocks whose working sets fit in specific cache levels (L1, L2, or L3). For conv2d, tiling across both spatial and channel dimensions enables temporal reuse of input/filter data within cache.

**Packing** complements tiling by reorganizing data layout so that elements accessed within a tile are contiguous in memory. Without packing, tiled accesses may have large strides that cause cache-line conflicts and TLB pressure. Packing converts strided accesses into sequential accesses within each tile, dramatically improving cache line utilization.

**Promotion** explicitly copies frequently reused tile data into stack-allocated buffers. This guarantees the working set resides in fast memory (L1/L2) and avoids interference from other data. It is particularly valuable when the same tile of input data is reused across multiple output channel computations.

### Intent 2: Compute Throughput and Vectorization (HIGH)

Once data is in cache, the bottleneck shifts to compute throughput. AVX2 provides 256-bit vector operations with FMA, offering up to 4x FP64 or 8x FP32 throughput over scalar code. Exploiting this requires careful loop structuring.

**Vectorization** maps the innermost loop iterations to SIMD vector lanes. The loop must have independent iterations and unit-stride memory access for efficient vector load/store. For conv2d, vectorizing along the output width or output channel dimension typically provides the cleanest mapping to AVX2 instructions.

**Loop Interchange** reorders the loop nest to place the most vectorization-friendly loop innermost and to ensure unit-stride access patterns in the innermost loop. It also affects data reuse patterns — the right loop order can dramatically reduce the number of cache misses by ensuring that reused data stays in registers or L1 cache across iterations of the outermost loops.

**Loop Unrolling** reduces loop overhead (branch prediction misses, induction variable updates) and exposes independent instructions to the out-of-order execution engine. Unrolling the innermost loop or a loop adjacent to the vectorized dimension enables software pipelining and register-level data reuse. On Broadwell, moderate unroll factors (2-8x) balance ILP gains against register pressure.

### Intent 3: Work Distribution and Iteration Space Restructuring (MEDIUM)

With 28 available cores, multi-core parallelism is essential for overall throughput. Additionally, structural transformations can fundamentally reshape the computation to enable more effective downstream optimizations.

**Parallelization** distributes iterations of outer parallel loops (batch, output channels, spatial) across CPU cores using OpenMP-style threading. The key concern is choosing the right loop level to parallelize — too fine-grained causes thread synchronization overhead, too coarse causes load imbalance. For a 28-core machine, distributing the batch or output-channel dimension typically provides sufficient parallelism with low overhead.

**Im2col Lowering** is a structural transformation that converts the convolution into a matrix multiplication (contraction) surrounded by data reorganization (reshape/gather) operations. This is significant because matmul is a far more regular computation with well-understood optimal schedules. After im2col, all subsequent optimizations (tiling, vectorization, packing) can leverage matmul-specific knowledge and achieve higher efficiency. The trade-off is additional memory for the im2col buffer.

**Peeling** separates boundary/remainder iterations from the main loop body. When iteration counts are not exact multiples of tile sizes or vector widths, the main loop body can be optimized aggressively (full tiles, aligned vector operations) while the peeled remainder handles edge cases with simpler scalar code. This is particularly important for spatial dimensions that may not be power-of-2 or vector-width-aligned.

## Action Selection Rationale

The 9 selected transformations cover the three fundamental performance axes for CPU loop-nest optimization:
1. **Memory hierarchy exploitation** (Tiling + Packing + Promotion)
2. **Compute unit utilization** (Vectorization + Loop Interchange + Loop Unrolling)
3. **Core-level parallelism and structural enablement** (Parallelization + Im2col Lowering + Peeling)

These transformations are composable — a typical high-performance schedule would involve im2col lowering (if applicable), followed by tiling, interchange, packing/promotion for cache optimization, vectorization for SIMD, unrolling for ILP, and parallelization for multi-core. Peeling handles any boundary conditions introduced by tiling or vectorization.

Transformations not included (Fusion, Padding, Loop Distribution, Canonicalization, Bufferization Strategy) are either less impactful for single-operation kernels (Fusion, Loop Distribution), secondary enabling transformations (Padding, Canonicalization), or lower-level concerns (Bufferization Strategy) that are better handled as automatic cleanup passes rather than explicit RL actions.
