# Action Enumeration Reasoning — v30 (dataset_conv2d, conv_2d_nchw_fchw)

## Workload Analysis

The benchmark set consists of 278 instances of `linalg.conv_2d_nchw_fchw`, a 2D convolution in NCHW input / FCHW filter layout. Each instance is a single operation with the computation pattern:

```
output[n, f, oh, ow] += input[n, c, oh*stride_h + kh, ow*stride_w + kw] * filter[f, c, kh, kw]
```

### Loop Nest Structure

This operation lowers to a **7-deep loop nest**:
- **4 parallel loops**: batch (N), output filters (F), output height (OH), output width (OW)
- **3 reduction loops**: input channels (C), kernel height (KH), kernel width (KW)

The total iteration count and data volume vary significantly across the 278 instances, requiring optimizations that generalize across shapes.

### Memory Access Pattern Characteristics

1. **Input tensor**: Accessed with a sliding window pattern. The base address depends on spatial output coordinates and kernel offsets. When kernel dimensions are > 1, the same input elements are reused across multiple output positions (spatial reuse). When stride > 1, the input access pattern skips elements, creating non-unit-stride access.

2. **Filter tensor**: Reused across all batch and spatial dimensions. For a fixed (f, c, kh, kw), the same filter value is used for every (n, oh, ow) combination. This makes the filter a strong candidate for cache-resident promotion.

3. **Output tensor**: Each element is accumulated over the full (C, KH, KW) reduction space. Keeping output tiles in registers or L1 cache during reduction is critical for write-back efficiency.

### Shape Diversity in the Benchmark Set

- **Batch sizes**: 128, 256 — large enough that batch-level parallelism alone can saturate 28 cores.
- **Channel dimensions**: 32 to 384 output filters, 48 to 288 input channels — reduction dimension sizes vary considerably.
- **Spatial sizes**: 7x7 to 28x28 input, with output as small as 3x3 and as large as 28x28 — some instances have very small spatial loops, limiting spatial tiling opportunities.
- **Kernel sizes**: Predominantly 1x1 and 3x3 — 1x1 kernels collapse the KH/KW reduction loops to trivial single iterations, effectively reducing the loop nest depth from 7 to 5.
- **Strides**: 1 or 2 — stride-2 convolutions halve spatial output dimensions and introduce non-unit stride in input access.

### Hardware Considerations (Intel Xeon E5-2680 v4, Broadwell)

- **28 physical cores**, 2 NUMA nodes — ample parallelism potential from batch and filter dimensions.
- **AVX2 + FMA**: 256-bit vectors → 4 FP64 lanes. Vectorizing the innermost loop along a contiguous dimension is essential.
- **Cache hierarchy**: L1d 32KB, L2 256KB, L3 ~35MB shared — multi-level tiling should target L1/L2 tile sizes.
- **No AVX-512**: Vector widths must assume 256-bit maximum.

## Optimization Strategy

### Intent 1: Data Locality and Cache Optimization (HIGH Priority)

This is the highest-impact intent. The 7-deep loop nest operates on three tensors with distinct reuse patterns. Without tiling, the combined working set of even moderate-sized instances (e.g., 128x128x14x14 input) far exceeds L1/L2 capacity. The convolution's sliding-window input access creates irregular (strided) memory patterns that further stress the cache hierarchy.

**Selected transformations:**

1. **Tiling** — The fundamental optimization for any deep loop nest. Multi-level tiling creates working sets that fit in L1/L2/L3. For the 7-loop conv2d nest, tiling along parallel dimensions (N, F, OH, OW) controls output tile size, while tiling reduction dimensions (C, KH, KW — primarily C, since KH/KW are often small) controls the reduction slice size. The product of tile dimensions determines the working set size for each cache level. Tiling is a prerequisite for most downstream optimizations (vectorization, promotion, parallelization).

2. **Loop Interchange** — The 7 loops have 7! = 5040 possible orderings (subject to legality). The default ordering may not be optimal for memory access patterns. For NCHW layout, the innermost storage dimension is W (width), so placing the OW loop innermost (for output writes) or arranging reduction loops to maximize spatial locality of input reads can dramatically reduce cache misses. Loop interchange is especially impactful for conv2d because the input access pattern depends on both spatial and kernel loop positions.

3. **Promotion** — After tiling, the tiled operand slices may still have non-unit strides due to the convolution's sliding-window access and the NCHW layout. Promoting (copying) tiled slices of input and filter into contiguous temporary buffers eliminates these irregular access patterns, enabling cleaner vectorization and reducing TLB pressure. Promotion is particularly valuable when kernel dimensions create overlapping input windows within a tile. Requires internal bufferization as a preprocessing step.

### Intent 2: Compute Throughput and Instruction Efficiency (HIGH Priority)

Once data is cache-resident through tiling and promotion, the next bottleneck is compute throughput. The convolution's multiply-accumulate operations map naturally to FMA instructions. Maximizing the utilization of AVX2 FMA units requires vectorizing innermost loops and exposing enough independent operations for the out-of-order engine.

**Selected transformations:**

1. **Vectorization** — Map iterations of an innermost loop to AVX2 vector lanes (4-wide for FP64). The choice of vectorized dimension affects performance: vectorizing along OW (output width) typically gives unit-stride output writes; vectorizing along C (channels) may give unit-stride filter access. The optimal choice depends on the loop order established by tiling and interchange. Vectorization is essential for achieving anywhere near peak throughput on this hardware.

2. **Unrolling** — After tiling creates small inner loop trip counts, unrolling the innermost loops (or the loop just above the vectorized loop) reduces branch overhead and exposes independent FMA operations to the CPU's instruction scheduler. For small tile sizes (e.g., 4x4 output tiles), full unrolling can eliminate loop control entirely. Unrolling also enables register-level accumulation of output elements across reduction iterations.

### Intent 3: Multi-Core Work Distribution (MEDIUM Priority)

The target has 28 physical cores across 2 NUMA nodes. The convolution's batch and filter dimensions provide natural parallelism — each (n, f) pair computes an independent output slice. For the batch sizes in this dataset (128, 256), there is more than enough work to distribute.

Priority is MEDIUM rather than HIGH because: (a) single-core performance from tiling + vectorization often dominates the speedup, and (b) excessive parallelization of small spatial tiles can introduce synchronization overhead that outweighs the parallel gain. Parallelization is most effective when applied to outer tiled loops after the tiling strategy is established.

**Selected transformations:**

1. **Parallelization (Tiling-based)** — Tile the outermost parallel loops and distribute tile iterations across threads. Tile sizes control the work granularity per thread. This approach is more flexible than direct parallelization because it naturally handles cases where loop bounds are not evenly divisible by thread count.

2. **Parallelization (Direct)** — Directly partition parallel loop iterations across a fixed number of threads. Simpler than tiling-based parallelization but requires iteration counts to be divisible by thread counts for correct downstream lowering. Most natural for the batch dimension where batch sizes (128, 256) divide evenly by common thread counts.

## Summary of Enumerated Actions

| # | Action | Intent | Priority |
|---|--------|--------|----------|
| 1 | Tiling | Data Locality | HIGH |
| 2 | Loop Interchange | Data Locality | HIGH |
| 3 | Promotion | Data Locality | HIGH |
| 4 | Vectorization | Compute Throughput | HIGH |
| 5 | Unrolling | Compute Throughput | HIGH |
| 6 | Parallelization (Tiling) | Work Distribution | MEDIUM |
| 7 | Parallelization (Direct) | Work Distribution | MEDIUM |

Total: 7 macro RL actions across 3 optimization intents.
