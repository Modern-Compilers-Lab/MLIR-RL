# Action Enumeration Reasoning — v49 (dataset_conv2d)

## Benchmark Analysis

**Operation**: `linalg.conv_2d_nchw_fchw` — 2D convolution with NCHW input layout and FCHW filter layout.

**Loop nest structure** (7-deep):
- Parallel loops: N (batch), F (output channels), OH (output height), OW (output width)
- Reduction loops: C (input channels), KH (kernel height), KW (kernel width)

**Data type**: f64 — on AVX2, this yields 4 SIMD lanes per 256-bit vector register.

**Shape diversity across 278 instances**:
- Batch sizes: 128 and 256
- Input channels (C): 32 to 288
- Spatial dimensions (H×W): 7×7 up to 56×56
- Output channels (F): 32 to 512
- Kernel sizes (KH×KW): 1×1, 3×3, and 7×7
- Strides: varying (derived from the ratio of input spatial size to output spatial size)
- Many instances use 1×1 kernels, which degenerate into batched matrix-multiply-like contractions (no spatial reduction over KH/KW).

**Target hardware**: Intel Xeon E5-2680 v4 (Broadwell), 28 physical cores, AVX2+FMA, no AVX-512. Cache hierarchy: 32KB L1d, 256KB L2, shared L3 (~35MB per socket).

## Reasoning by Optimization Intent

### Intent 1: Data Locality and Cache Optimization (HIGH priority)

This is the highest-impact optimization category for conv2D on CPU. The 7-deep loop nest accesses three large tensors (input, filter, output) with different reuse patterns:
- **Filter reuse**: The filter tensor is reused across all batch elements and all output spatial positions. If the filter tile fits in L1/L2, it can be reused extensively.
- **Input reuse**: The input tensor is reused across output channels (F dimension). Overlapping windows in spatial dimensions also create reuse across adjacent output positions.
- **Output reuse**: The output tensor is accumulated over all reduction dimensions (C, KH, KW).

Without tiling, working sets far exceed cache capacity (e.g., a single 128×128×14×14×8B input tensor is ~25MB). Multi-level tiling is essential.

**Tiling**: The fundamental transformation. Partitions the iteration space into blocks that fit cache levels. For a 7-deep loop nest, tile sizes along different dimensions control which data reuse pattern is exploited. This is the single most impactful action for cache performance.

**Loop Interchange**: The default loop ordering may not maximize spatial locality given the NCHW memory layout (W-contiguous). Reordering loops so that the innermost iteration sweeps over contiguous memory dimensions (W for input, KW for filter) reduces cache misses and enables hardware prefetching. Also affects which type of reuse (temporal vs. spatial) is prioritized.

**Promotion**: After tiling, the operand slices accessed within a tile may still have strided memory access patterns (due to the NCHW layout, the tiled C-dimension slices are not contiguous). Promotion copies these slices into contiguous scratch buffers, converting strided accesses into dense sequential accesses. This is particularly impactful for filter tiles where the FCHW layout means C-slices are non-contiguous when tiling along F. Requires bufferization as a preprocessing step.

**Im2col Lowering**: Converts the convolution into a matmul-like contraction surrounded by reshape operations. This fundamentally regularizes the access pattern — the irregular overlapping-window access of convolution becomes a single dense matrix multiply. This is especially beneficial for non-1×1 kernels where the convolution access pattern involves strided, overlapping spatial windows. After im2col, all subsequent optimizations (tiling, vectorization, etc.) target the resulting contraction op, which has much simpler and more regular access patterns. For 1×1 kernels, this is less impactful since the convolution is already essentially a matrix multiply.

### Intent 2: Compute Throughput and Parallelism (HIGH priority)

Conv2D is compute-intensive (O(N·F·OH·OW·C·KH·KW) FMA operations). On Broadwell with AVX2, each core can execute one 256-bit FMA per cycle, giving 4 FP64 FMAs/cycle. Without vectorization, only 1 FMA/cycle is used — a 4x underutilization of peak throughput. Additionally, the convolution has four fully parallel loop dimensions (N, F, OH, OW) with no loop-carried dependencies, making thread-level distribution across 28 cores essential for full hardware utilization. The benefit of parallelism is shape-dependent: small spatial dimensions (7×7) may provide insufficient work per thread, while large tensors may become memory-bandwidth-bound.

**Vectorization (Sequential Tiling Preprocessing)**: Tile the innermost loops to match vector widths (multiples of 4 for FP64) using sequential `tile_using_for`, then lower the tiled inner loops to SIMD vector operations. The sequential tiling ensures the original loop structure is preserved with an inner tile that maps directly to vector registers. This is the standard approach when parallelism is handled separately at a different loop level.

**Vectorization (Parallel Tiling Preprocessing)**: Tile the innermost loops to vector widths using `tile_using_forall`, which creates parallel outer tiles that can be distributed across threads. The inner tile is then vectorized. This combines vectorization with coarse-grain parallelism in a single step, which can be more efficient than applying them separately because it avoids redundant loop transformations and naturally aligns parallel granularity with vector granularity.

**Parallelization (Tiling-based)**: Tile the parallel loop dimensions and distribute the resulting tiles across threads. This provides explicit control over the granularity of parallel work (tile size determines how much work each thread gets). Tiling-based parallelization integrates naturally with the cache tiling strategy — outer tiles can be distributed while inner tiles target cache locality.

**Parallelization (Thread-based)**: Directly map parallel loop iterations to a specified number of threads without explicit tiling. This is simpler but requires the iteration count to be divisible by the thread count. More suitable for loops where the iteration count is already well-matched to the available cores (e.g., batch dimension of 128 divided across 28 cores, though not evenly).
