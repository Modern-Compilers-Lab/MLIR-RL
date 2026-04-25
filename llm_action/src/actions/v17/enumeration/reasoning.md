# Action Enumeration Reasoning — v17 (2D Convolution, NCHW/FCHW)

## Input Analysis

The input is a `linalg.conv_2d_nchw_fchw` operation representing a 2D convolution with NCHW input layout and FCHW filter layout. The concrete instance has:
- Input: `128x32x7x7` (N=128, C=32, H=7, W=7)
- Filter: `256x32x1x1` (F=256, C=32, KH=1, KW=1)
- Output: `128x256x7x7` (N=128, F=256, OH=7, OW=7)

From a loop-nest perspective, this operation corresponds to a 7-deep nested loop with:
- **Parallel dimensions**: N (batch), F (output channels), OH (output height), OW (output width)
- **Reduction dimensions**: C (input channels), KH (kernel height), KW (kernel width)

For the 1x1 kernel case, the KH and KW loops are trivially single-iteration, making the computation effectively a batched contraction over the channel dimension — structurally similar to a batched matrix multiplication.

## Target Hardware Considerations

- **Intel Xeon E5-2680 v4 (Broadwell)**: 28 physical cores, 2 NUMA nodes, no hyperthreading
- **AVX2 + FMA**: 4 FP64 lanes (256-bit vectors), no AVX-512
- **Cache hierarchy**: L1d 32KB/core, L2 256KB/core, shared L3 per socket (~35MB)
- Optimization emphasis: cache-aware tiling, SIMD vectorization, coarse-grain parallelism

## Working Set Analysis

The total data footprint for the concrete instance:
- Input: 128 * 32 * 7 * 7 * 8B = ~1.57 MB
- Filter: 256 * 32 * 1 * 1 * 8B = ~64 KB
- Output: 128 * 256 * 7 * 7 * 8B = ~12.5 MB

The combined working set (~14 MB) far exceeds L1/L2 capacity and approaches L3 capacity. Multi-level tiling is essential to create reusable data blocks that fit within cache.

## Optimization Intent Reasoning

### Intent 1: Memory Hierarchy Optimization (HIGH Priority)

This is the most impactful optimization category for loop-nest computations with large working sets. Three complementary transformations address different aspects:

1. **Tiling**: Partitions the iteration space so each tile's working set fits in L1 or L2 cache. This is the single most critical transformation for dense loop nests — without it, every iteration fetches from main memory.

2. **Loop Interchange**: Reorders loops to place stride-1 accesses innermost. For NCHW layout, the W dimension is stride-1 for the input tensor, while the OW dimension is stride-1 for the output. Correct loop ordering enables hardware prefetching and avoids cache-line-level thrashing.

3. **Promotion**: After tiling, tiles may still suffer from non-contiguous memory access patterns due to the multi-dimensional layout. Copying (packing) tiles into contiguous temporary buffers eliminates conflict misses and stride irregularities, maximizing effective cache utilization. This is especially important when the inner tile dimensions don't align with the tensor's storage layout.

### Intent 2: SIMD and Instruction-Level Parallelism (HIGH Priority)

Peak throughput on Broadwell requires filling AVX2 vector lanes and keeping the FMA pipeline busy:

1. **Vectorization**: Maps loop iterations to SIMD lanes. With AVX2, 4 FP64 elements can be processed per instruction. The choice of which loop dimension to vectorize (output spatial, output channels, or reduction channels) significantly impacts performance and depends on the tiled loop structure.

2. **Unrolling**: After vectorization, unrolling an adjacent loop dimension exposes multiple independent FMA operations to the out-of-order engine. This hides latency (FMA has ~5 cycle latency on Broadwell) and achieves higher utilization of the two FMA execution units per core.

### Intent 3: Coarse-Grain Parallelism and Loop Structure (MEDIUM Priority)

With 28 cores available, thread-level parallelism is necessary for wall-clock performance, and iteration-space restructuring can unlock further optimization potential:

1. **Parallelization**: The parallel dimensions (N, F, OH, OW) provide ample parallelism. Distributing outer tiled loops across threads is straightforward and yields near-linear scaling, provided tiles are large enough to amortize synchronization overhead and NUMA-aware placement is considered.

2. **Kernel Lowering (im2col)**: Convolution's sliding-window access pattern creates irregular memory access for general kernel sizes. Im2col lowering reshapes the input data and converts the convolution into a standard matrix-matrix contraction, which has perfectly regular access patterns. This makes subsequent tiling, vectorization, and packing more effective by operating on a canonical contraction loop nest. For the 1x1 kernel case this is less critical (the structure is already contraction-like), but for general kernel sizes it is a key enabler.

## Priority Justification

- **Memory Hierarchy Optimization (HIGH)**: Without tiling and data layout optimization, the working set vastly exceeds cache, making all other optimizations secondary. Tiling alone can yield 5-20x improvement for large loop nests.
- **SIMD and ILP (HIGH)**: Vectorization provides up to 4x throughput improvement on FP64, and unrolling further improves pipeline utilization. These are essential for approaching peak FLOP/s.
- **Coarse-Grain Parallelism and Loop Structure (MEDIUM)**: Thread parallelism provides linear scaling with core count but depends on the tiled loop structure established by the first two intents. Kernel lowering is shape-dependent — critical for large kernel sizes but less impactful for 1x1 kernels.
