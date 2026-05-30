# Action Enumeration Reasoning — v45

## Benchmark Analysis

The dataset_ml training set contains 1135 instances across five kernel families:

1. **conv_2d_nchw_fchw** (277 instances): 2D convolution in NCHW/FCHW layout. These are compute-intensive operations with deep loop nests (7 loops: batch, output channels, output height, output width, input channels, kernel height, kernel width). Shapes range from small spatial extents (7x7) to large (150x150), with varying channel counts (32–512) and kernel sizes (1x1 to 7x7). Strides vary (1 or 2). The irregular loop nest structure makes direct optimization challenging — im2col lowering to matmul-like contraction is a standard strategy.

2. **matmul** (186 instances): Classic matrix multiplication with I×J @ J×K dimensions. Three nested loops with one reduction dimension. Sizes range from 128 to 3072 per dimension. This is the canonical compute-bound kernel where tiling, packing, vectorization, and parallelization have well-understood, high-impact effects.

3. **pooling_nchw_max** (249 instances): Max pooling over spatial windows. Four outer loops (batch, channel, output height, output width) with two reduction loops (kernel height, kernel width). Memory-access-intensive with reduction semantics. Window sizes range from 1x1 to 7x7, spatial dimensions from 5x5 to 150x150. Benefits from tiling and vectorization, though the reduction (max) limits some vectorization strategies.

4. **add** (270 instances): Elementwise 4D tensor addition. Four parallel loops with no reductions. Purely memory-bandwidth bound — performance is dominated by data movement, not compute. Shapes are highly varied. Vectorization and parallelization are the primary levers; tiling helps cache blocking for large tensors.

5. **relu** (148 instances): Elementwise ReLU (compare + select). Two or four parallel loops (instances show both 2D and 4D shapes). Like add, this is memory-bandwidth bound. Same optimization strategy as add applies.

## Hardware Considerations

Target: Intel Xeon E5-2680 v4 (Broadwell)
- 28 physical cores, 2 sockets × 14 cores, 2 NUMA nodes
- AVX2 + FMA (no AVX-512): FP64 has 4 vector lanes (256-bit)
- Cache: L1d ~32KB, L2 ~256KB per core, shared L3 per socket
- No hyperthreading

Key implications:
- **Cache tiling is critical**: Working sets for matmul/conv easily exceed L1/L2. Tiling to fit L1 (32KB) or L2 (256KB) is essential.
- **AVX2 vectorization**: 4-wide FP64 SIMD. Vectorizing innermost loops gives 4× throughput for compute-bound ops and saturates bandwidth for memory-bound ops.
- **28-way parallelism**: Outer loop parallelization across 28 cores is necessary for large problems. Must avoid oversubscription.
- **Register pressure**: Broadwell has 16 YMM registers. Aggressive unrolling can cause spills — moderate unroll factors (2–8) are safer.

## Intent and Transformation Reasoning

### Intent 1: Data Locality and Cache Optimization (HIGH priority)

This is the highest-impact optimization category for ALL kernel types in the set.

- **Matmul**: Classic example where tiling transforms O(N³) cache misses to O(N³/√M) with optimal blocking. Packing eliminates TLB and stride-related misses in tiled blocks.
- **Conv2d**: Even larger iteration spaces (7 loops). Tiling the output spatial and channel dimensions to fit cache, combined with packing, is essential.
- **Pooling**: Moderate compute, but spatial access patterns benefit from cache-friendly tiling.
- **Add/ReLU**: Memory-bound, but tiling ensures streaming through cache lines efficiently rather than thrashing across large tensor dimensions.

Transformations:
1. **Tiling**: The foundational transformation. Partitions the iteration space into blocks that fit in L1/L2 cache. Applicable to all kernel types. For matmul, standard 3-level tiling (I, J, K dimensions). For conv, tiling output spatial + channel dimensions.
2. **Loop Interchange**: Reorders loop dimensions to improve spatial locality (stride-1 access on innermost loop) and temporal reuse (reuse-carrying loops moved inward). Critical for conv where the default loop order may not be cache-optimal.
3. **Packing**: Copies tiled data into contiguous, stride-free buffers. Eliminates non-unit-stride access and TLB thrashing within tiles. Most impactful for matmul and conv where operand layouts cause strided access in inner loops.
4. **Promotion**: Promotes tiled operands into local contiguous buffers at the buffer/memref level. Requires bufferization as a preprocessing step, followed by canonicalization to fold dynamic shapes into static types. Amortizes copy cost over inner tile iterations.

### Intent 2: SIMD Vectorization and Compute Throughput (HIGH priority)

AVX2 + FMA provides 4× FP64 throughput per cycle. This directly multiplies peak FLOPS for compute-bound kernels and saturates memory bandwidth with fewer instructions for memory-bound kernels.

Transformations:
1. **Vectorization (Sequential)**: Tiles innermost loops to match vector width, then lowers to SIMD operations. The preprocessing tiling uses sequential `for` loops (`tile_using_for`). This is the standard vectorization approach — straightforward and always applicable when loop bounds are compatible.
2. **Vectorization (Parallel)**: Same SIMD lowering but preprocessing tiling uses parallel `forall` loops (`tile_using_forall`), which also distributes outer tiles across threads. Combines vectorization with parallelization in one transformation — particularly effective for elementwise ops (add, relu) where the entire iteration space is parallel.
3. **Loop Unrolling**: Reduces loop overhead (branch prediction, counter updates) and exposes instruction-level parallelism for out-of-order execution and FMA pipelining. Most effective on innermost loops after tiling, with moderate unroll factors (2–8) to avoid register pressure on Broadwell's 16 YMM registers.

### Intent 3: Thread-Level Parallelism and Iteration Space Restructuring (HIGH priority)

With 28 physical cores available, coarse-grain parallelism is essential for utilizing the hardware. Additionally, restructuring the iteration space (im2col for convolution) simplifies the loop nest structure, enabling all downstream optimizations to work more effectively.

Transformations:
1. **Parallelization (Tiling)**: Tiles outer parallel dimensions into chunks and distributes tiles across threads. Reduction dimensions are automatically excluded. Provides fine control over work granularity — tile sizes can be tuned to balance load across 28 cores while maintaining cache locality per thread.
2. **Parallelization (Threads)**: Directly distributes loop iterations across a fixed thread pool (e.g., 14 or 28 threads). Simpler than tiling-based parallelization but requires iteration count divisibility. Better for regular, balanced workloads like elementwise operations.
3. **Im2col Lowering**: Converts conv_2d_nchw_fchw into a matmul-like contraction (the primary compute op) surrounded by reshape operations. This is a kernel-specific but reusable iteration space transformation that collapses the 7-loop convolution nest into a cleaner contraction form. All subsequent optimizations (tiling, vectorization, parallelization) target the resulting contraction, not the reshapes. Unlocks the full matmul optimization stack for convolution kernels, which is critical given conv2d represents 24% of the dataset.

## Summary

| Intent | Priority | Transformations | Primary Beneficiaries |
|--------|----------|-----------------|-----------------------|
| Data Locality and Cache Optimization | HIGH | Tiling, Loop Interchange, Packing, Promotion | matmul, conv2d, pooling |
| SIMD Vectorization and Compute Throughput | HIGH | Vectorization (Sequential), Vectorization (Parallel), Loop Unrolling | all kernels |
| Thread-Level Parallelism and Iteration Space Restructuring | HIGH | Parallelization (Tiling), Parallelization (Threads), Im2col Lowering | all kernels (im2col: conv2d only) |

Total: 3 intents, 10 unique transformations covering the complete optimization stack for CPU-targeted ML kernels.
