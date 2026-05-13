# Action Enumeration Reasoning — v28 (paper_conv2d)

## Benchmark Set: paper_conv2d (train split, 3 instances)

All instances are `linalg.conv_2d_nchw_fchw` with `dilations=1, strides=1`.

## Loop Nest Structure

The `linalg.conv_2d_nchw_fchw` operation forms a 7-dimensional loop nest:
- **4 parallel loops**: N (batch), F (output filters), OH (output height), OW (output width)
- **3 reduction loops**: C (input channels), KH (kernel height), KW (kernel width)

Access patterns:
- Input: `tensor[N, C, OH+KH, OW+KW]` — strided windowed access over spatial dimensions
- Filter: `tensor[F, C, KH, KW]` — fully traversed per output point (high reuse potential)
- Output: `tensor[N, F, OH, OW]` — each element written once (accumulation over reduction dims)

## Shape Analysis

| Instance | N | C | H | W | F | KH | KW | OH | OW | Character |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 128 | 28 | 28 | 512 | 1 | 1 | 28 | 28 | 1x1 conv (degenerate kernel, effectively batched matmul) |
| 2 | 256 | 512 | 28 | 28 | 128 | 1 | 1 | 28 | 28 | 1x1 conv (reversed channel ratio) |
| 3 | 256 | 64 | 56 | 56 | 64 | 3 | 3 | 54 | 54 | Standard 3x3 conv with non-trivial kernel |

### Key observations:
- **1x1 convolutions** (instances 1 and 2): KH=KW=1, so kernel spatial reduction loops are trivial (trip count 1). The computation is essentially a batched matrix multiplication across all spatial positions. Reduction is only over C.
- **3x3 convolution** (instance 3): Has full kernel spatial loops (KH=KW=3). The sliding-window access pattern over input creates strided, non-contiguous memory access — a key bottleneck.
- **Large batch dimension** (N=256): Provides ample parallelism for thread distribution.
- **Tensor sizes are very large**: e.g., 256x128x28x28 FP64 ~ 200MB. Working sets far exceed all cache levels.

## Hardware Considerations (Intel Xeon E5-2680 v4 — Broadwell)

- **28 physical cores**, 2 NUMA nodes, no HT
- **AVX2 + FMA**: 4 FP64 lanes per 256-bit vector, no AVX-512
- **Cache hierarchy**: L1d=32KB, L2=256KB, L3=~35MB shared per socket
- FP64 element size: 8 bytes; 4 elements per AVX2 vector

## Optimization Analysis

### 1. Cache Locality is Critical (HIGH priority)

The input tensors are hundreds of megabytes. Without tiling, even L3 is insufficient. The filter tensor has high temporal reuse: it is accessed for every (N, OH, OW) combination. Tiling to fit filter tiles + input/output tiles in L2 (256KB) can dramatically reduce memory traffic.

Loop interchange matters because NCHW layout means W is the fastest-varying dimension. The default loop order may not align well with all three tensor layouts. Reordering to place contiguous-access dimensions innermost reduces cache line waste.

Promotion of tiled operands into contiguous temporary buffers eliminates strided access within tiles. The filter tensor particularly benefits: after tiling over C and F, the filter sub-tensor may have large strides from the original allocation.

Packing (data layout transformation at tensor level) complements tiling by rearranging the physical data layout to match the tiled iteration structure. This ensures stride-1 access on the innermost dimension within each tile.

### 2. SIMD Exploitation is Essential (HIGH priority)

Without vectorization, only 1 FP64 operation per cycle vs. 4 with AVX2 FMA. The innermost spatial dimension (OW=28-54) provides enough contiguous elements for multiple 4-wide vector operations.

Unrolling inner loops exposes multiple independent FMA operations to fill the pipeline (FMA latency is multiple cycles; need independent ops to keep the pipeline busy).

Im2col lowering is particularly relevant for the 3x3 conv instance: it converts the irregular windowed access pattern (input[n, c, oh+kh, ow+kw]) into a contiguous matmul-like contraction. This eliminates stride irregularity and produces a dense loop nest that vectorizes cleanly. For 1x1 convs, the benefit is marginal since the access pattern is already regular.

### 3. Thread-Level Parallelism is Necessary (HIGH priority)

With 28 cores available and N=256, there are far more independent iterations than cores across the parallel dimensions. Two complementary strategies:
- **Tiling-based**: Creates tile-sized work units, flexible for non-divisible dimensions
- **Direct thread mapping**: Simpler when dimensions divide evenly (N=256 is divisible by 2,4,8,16)

Peeling after tiling (for cache or parallelism) is important when dimensions don't divide evenly by tile sizes. It separates remainder iterations into a cleanup loop, allowing the main body to assume uniform tile sizes for more aggressive optimization (full vectors, uniform work per thread).

## Transformation Summary

10 distinct RL macro actions organized under 3 intents:

1. **Cache Locality and Data Reuse** (HIGH): Tiling, Loop Interchange, Promotion, Packing
2. **SIMD Exploitation and ILP** (HIGH): Vectorization, Unrolling, Im2col Lowering
3. **Thread-Level Parallelism** (HIGH): Parallelization Tiling, Parallelization Direct, Peeling

All intents are HIGH priority because:
- The tensor sizes make cache optimization non-optional
- AVX2 FMA provides 4x throughput that is wasted without vectorization
- 28 idle cores represent 96.4% wasted compute without parallelization
