# Action Enumeration Reasoning — v20 (2D Convolution, NCHW/FCHW)

## Input Analysis

The input is a `linalg.conv_2d_nchw_fchw` operation — a 2D convolution with:
- Input tensor: `[N x C x H x W]` (batch, input channels, height, width)
- Filter tensor: `[F x C x KH x KW]` (output channels, input channels, kernel height, kernel width)
- Output tensor: `[N x F x OH x OW]` (batch, output channels, output height, output width)

From a loop-nest perspective, this represents a **7-deep nested loop**:
- **Parallel loops**: N, F, OH, OW (output indices)
- **Reduction loops**: C, KH, KW (input channel and kernel spatial indices)

The computation pattern is:
```
for n, f, oh, ow (parallel):
  for c, kh, kw (reduction):
    output[n,f,oh,ow] += input[n,c,oh+kh,ow+kw] * filter[f,c,kh,kw]
```

## Hardware Context

Target: Intel Xeon E5-2680 v4 (Broadwell)
- 28 physical cores (2 sockets, 14 cores each)
- AVX2 + FMA (256-bit vectors, 4 FP64 lanes)
- L1d: 32KB, L2: 256KB, shared L3 per socket
- No AVX-512

## Optimization Reasoning

### Intent 1: Cache Locality & Data Reuse (HIGH priority)

This is the most critical optimization dimension for convolutions. The 7-deep loop nest has complex data reuse patterns:
- The input tensor has overlapping accesses due to the sliding window (OH+KH, OW+KW indexing).
- The filter tensor is reused across all spatial positions and batch elements.
- The output tensor has temporal reuse across the reduction loops (C, KH, KW).

Without tiling, the working set far exceeds L1/L2 capacity for realistic sizes. Tiling partitions the iteration space so that tiles of input, filter, and output fit in cache. Loop interchange can improve stride-1 access patterns. Promotion copies tiled operands into contiguous scratch buffers, eliminating stride issues from the original layout and enabling efficient vectorized loads.

### Intent 2: SIMD Exploitation (HIGH priority)

AVX2 provides 4 FP64 FMA operations per cycle per core. The innermost loop(s) must be mapped to vector lanes. For convolution:
- Vectorizing across the output-width or output-channel dimension gives stride-1 output writes.
- Unrolling exposes independent multiply-accumulate operations that the FMA units can pipeline.

Vectorization is essential because without it, the code uses scalar arithmetic at 1/4 of potential throughput.

### Intent 3: Parallelism & Iteration Space Restructuring (MEDIUM priority)

With 28 cores available, the outer parallel loops (N, F, OH, OW) provide ample parallelism for most practical shapes. Parallelization distributes work across cores.

Additionally, im2col lowering restructures the convolution into a matrix multiplication by materializing the sliding-window input patches as a contiguous matrix. This transforms the 7-deep irregular-access loop nest into a regular contraction (matmul), which benefits from decades of optimized BLAS-style scheduling. This is particularly valuable for small kernel sizes (e.g., 1x1 or 3x3) where the reduction loop trip counts are short and vectorization of the original conv loop nest is less effective.

## Action Count Rationale

- 3 intents covering the three major performance axes: memory hierarchy, instruction-level parallelism (SIMD), and thread-level parallelism.
- 9 total transformations providing comprehensive coverage of standard HPC loop-nest optimizations applicable to convolution workloads on CPU.
