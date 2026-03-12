# Layer 1 — Action Enumeration Reasoning (v3)

## Analysis of Input Operations

The input MLIR templates cover three representative kernel types:

1. **Matrix Multiplication** (`linalg.matmul`): A 3-deep loop nest (I, J, K) with two parallel dimensions (I, K) and one reduction dimension (J). Memory access patterns involve row-major reads on the left operand, column-strided reads on the right operand, and row-major writes on the output. This is a classic compute-bound kernel where data reuse is critical.

2. **2D Convolution** (`linalg.conv_2d_nchw_fchw`): A 7-deep loop nest (N, F, OH, OW, C, KH, KW) with four parallel dimensions (N, F, OH, OW) and three reduction dimensions (C, KH, KW). The memory access pattern involves sliding windows over spatial dimensions with channel-wise reductions. This kernel has complex reuse patterns across filter, spatial, and channel dimensions.

3. **Generic Element-wise** (`linalg.generic` with 5 parallel dims): A 5-deep loop nest with all parallel dimensions and element-wise memory access (identity indexing maps). This is memory-bandwidth-bound with no data reuse across iterations; performance hinges on efficient traversal and vectorization.

## Hardware Context (Intel Xeon E5-2680 v4, Broadwell)

- **AVX2 + FMA**: 256-bit vectors; FP64 = 4 lanes, FP32 = 8 lanes.
- **Cache hierarchy**: L1d=32KB, L2=256KB, L3 shared per socket (~35MB).
- **28 cores** across 2 NUMA nodes, no SMT.
- Key performance levers: cache-aware tiling, SIMD vectorization, coarse-grain parallelism, and NUMA-aware data placement.

## Optimization Intent Selection Rationale

### Intent 1: Data Locality and Cache Utilization (HIGH)
For compute-bound kernels (matmul, convolution), the dominant performance bottleneck is moving data through the memory hierarchy. Tiling to fit working sets into L1/L2 caches is the single most impactful optimization. Packing/promotion of tiles into contiguous buffers further improves cache line utilization by eliminating stride-related conflicts. These are the foundational transformations around which all other optimizations compose.

### Intent 2: SIMD Exploitation (HIGH)
AVX2+FMA provides up to 4 FP64 FLOPs per cycle per lane (8 FLOPs with FMA). Without vectorization, the kernel runs at 1/4 or 1/8 of peak throughput. Vectorization of innermost loops, combined with loop interchange to place contiguous-memory dimensions innermost, is essential. Unrolling further exposes independent instructions to fill FMA pipeline latency.

### Intent 3: Parallelism and Work Distribution (MEDIUM)
With 28 cores available, parallel distribution of outer loop iterations is important for large problems. However, the benefit is shape-dependent (small problems may not benefit from full parallelization) and requires care to avoid cache thrashing and NUMA penalties. Loop distribution can also enable partial parallelization of otherwise sequential loop nests.

### Intent 4: Iteration Space Restructuring (MEDIUM)
Transformations that restructure the iteration space — interchange for better memory access order, fusion for producer-consumer locality, fission for enabling other transformations — are enabling optimizations. They are rarely sufficient alone but unlock the effectiveness of tiling, vectorization, and parallelization. For convolution, lowering to matmul-like forms can expose more regular loop nests amenable to standard optimizations.

## Transformation Selection Principles

- Each transformation is a single, reusable macro RL action.
- Transformations are kernel-agnostic and dimension-agnostic.
- Parameters (which loops, what sizes, etc.) are deferred to Layer 2.
- No compound actions: each transformation does one thing.
- Names use canonical noun form.
