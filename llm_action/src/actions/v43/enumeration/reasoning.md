# Action Enumeration Reasoning — conv_2d_nchw_fchw (v43)

## Workload Analysis

The benchmark set consists of 278 `conv_2d_nchw_fchw` instances with f64 data type. The operation computes a 2D convolution with NCHW input layout and FCHW filter layout.

### Loop Nest Structure

`linalg.conv_2d_nchw_fchw` represents a 7-dimensional loop nest:
- **Parallel dimensions**: N (batch), F (output channels), OH (output height), OW (output width)
- **Reduction dimensions**: C (input channels), KH (kernel height), KW (kernel width)

Compute complexity: O(N * F * C * OH * OW * KH * KW)
Data volume: O(N*C*H*W + F*C*KH*KW + N*F*OH*OW)

The arithmetic intensity (compute/data) is high, especially for large channel counts and spatial dimensions, meaning the workload is compute-bound once data is in cache — but memory-bound if cache is not managed.

### Shape Characteristics

From the sampled shapes:
- **Batch sizes**: predominantly 128, some 256
- **Input channels**: 32 to 384
- **Output channels**: 32 to 512
- **Input spatial**: from 7x7 to 130x130
- **Kernel sizes**: predominantly 1x1, some 3x3
- **Output spatial**: from 3x3 to 65x65
- **Strides**: 1 or 2 (derived from input-output spatial relationship)

Key observations:
1. **1x1 kernels dominate**: Many instances have KH=KW=1, which collapses the kernel reduction loops and effectively turns the convolution into a batched matrix multiplication over spatial positions. This is significant because it simplifies the loop nest to 5 effective dimensions.
2. **Large batch sizes**: N=128 or N=256 means the batch dimension offers substantial parallelism and tiling opportunity.
3. **Varied spatial sizes**: Output spatial ranges from 3x3 (very small, few iterations) to 65x65 (substantial iteration count). This affects tiling strategies.
4. **f64 data type**: 8 bytes per element means AVX2 provides 4 lanes per vector. Cache footprints are 2x larger than f32 workloads.

### Hardware Mapping Considerations (Intel Xeon E5-2680 v4)

- **28 cores, 2 NUMA nodes**: Outer parallel loops (N, F) can be distributed across cores.
- **AVX2 with FMA**: 4 f64 elements per SIMD vector (256-bit). Vectorization of the innermost loop is essential.
- **Cache hierarchy**: L1=32KB, L2=256KB per core. For f64, L1 holds ~4096 elements, L2 holds ~32768 elements. Tile sizes must keep working sets within these limits.
- **No AVX-512**: Cannot rely on 512-bit vectors or masking.

## Optimization Strategy

### Intent 1: Cache Locality and Data Reuse (HIGH priority)

The 7-dimensional loop nest with large tensors (batch=128-256, channels up to 384-512) creates working sets far exceeding cache capacity. Without tiling, each element of the input tensor may be fetched from main memory multiple times across the reduction loops. Tiling, loop interchange, promotion, and packing collectively ensure data reuse within cache.

- **Tiling**: The foundational transformation. Partitions the 7-dimensional iteration space into blocks where the working set (slices of input, filter, output) fits in L1 or L2. For conv2d, tiling along output channels (F), input channels (C), and spatial dimensions (OH, OW) are all beneficial.
- **Loop Interchange**: The default loop ordering may cause poor stride patterns. For NCHW layout, accessing input along H/W dimensions within a channel is stride-1, but accessing across channels is strided. Reordering can improve locality, especially placing reduction dimensions (C, KH, KW) in positions that maximize reuse of loaded data.
- **Promotion**: After tiling, operand sub-tiles (especially filter tiles in FCHW layout) may still be accessed with non-unit stride. Copying them into compact contiguous buffers enables stride-1 vector loads and eliminates TLB pressure.
- **Packing**: Reorganizes the global data layout into a blocked format matching the tiling structure, so that tile-sized sub-tensors are stored contiguously. This is particularly valuable for the filter tensor where accessing a tile of F output channels requires gathering non-contiguous memory.

### Intent 2: SIMD and Instruction-Level Parallelism (HIGH priority)

Peak f64 throughput on Broadwell requires full utilization of AVX2 FMA units. Without vectorization, only scalar FMA is used — 1/4 of peak FLOPS. The innermost loop must be mapped to SIMD lanes.

- **Vectorization (Sequential Preprocessing)**: Tiles the innermost loops sequentially (tile_using_for) to match the AVX2 vector width (4 for f64), then lowers to vector operations. The outer tiles remain sequential — parallelism is handled by a separate parallelization action.
- **Vectorization (Parallel Preprocessing)**: Same vectorization but uses tile_using_forall for preprocessing, which simultaneously distributes outer tiles across threads. More aggressive; combines SIMD and parallelism.
- **Loop Unrolling**: After vectorization, the inner loop may be latency-bound (FMA latency ~4-5 cycles on Broadwell). Unrolling by a small factor exposes multiple independent computation chains to the out-of-order engine, hiding FMA latency and reducing loop overhead.

### Intent 3: Iteration Space Restructuring and Work Distribution (MEDIUM priority)

Conv2d has a complex 7-dimensional loop nest. Restructuring can simplify the computation (im2col converts it to a contraction), and work distribution across 28 cores provides throughput scaling.

- **Im2col Lowering**: Transforms the convolution into a matmul-like contraction by explicitly materializing the input patches. The resulting contraction has a simpler loop nest that is more amenable to standard tiling and vectorization. Particularly relevant for non-1x1 kernels where the kernel spatial reduction creates complex access patterns.
- **Parallelization (Tiling-Based)**: Partitions the iteration space into coarse tiles distributed across threads. Flexible for non-divisible dimension sizes. Can target the batch or output channel dimensions for independent parallel work.
- **Parallelization (Thread-Count-Based)**: Directly distributes iterations across a fixed thread count. Simpler and lower overhead when dimension sizes divide evenly by thread count (which is common for batch sizes of 128, 256).
