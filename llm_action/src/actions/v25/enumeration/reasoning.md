# Action Enumeration Reasoning — v25

## Benchmark Set Analysis

The training benchmark set ("sample", 20 instances) covers five operation families:

### 1. Matrix Multiplication (`linalg.matmul`)
- 3 nested loops: 2 parallel (I, K) and 1 reduction (J).
- Shapes range from 128x256x128 to 1024x128x1024. These are compute-bound with O(I*J*K) FLOPs and O(I*J + J*K + I*K) memory footprint.
- Classical target for tiling (L1/L2 blocking), vectorization (innermost accumulation), and parallelization (outer dimensions).

### 2. 2D Convolution (`linalg.conv_2d_nchw_fchw`)
- 7 nested loops: 4 parallel (N, F, OH, OW) and 3 reduction (C, KH, KW).
- Large batch sizes (128, 256), varied channel/spatial dimensions. Strides of 1 and 2 present.
- Deep loop nest benefits strongly from tiling, loop interchange (to improve reduction loop access patterns), and im2col lowering (converting to matmul-like contraction for standard optimization).

### 3. Pooling (`linalg.pooling_nchw_max`)
- 6 nested loops: 4 parallel (N, C, OH, OW) and 2 reduction (KH, KW).
- Similar structure to convolution but with max reduction instead of multiply-accumulate. Smaller reduction dimensions (kernel sizes 1x1 and 3x3).
- Benefits from tiling, vectorization of the max-reduction, and parallelization over batch/channel/spatial.

### 4. Element-wise Addition (`linalg.add`)
- 4 parallel loops, no reductions. Purely memory-bound.
- Shapes like 112x112x14x15, 112x224x56x130 — large total element counts.
- Primary optimization levers: vectorization (4 FP64 lanes via AVX2), parallelization (28 cores), and tiling for cache-line reuse.

### 5. ReLU (`linalg.generic` with element-wise max(x, 0))
- 2-4 parallel loops, no reductions. Memory-bound.
- Shapes include 2D (128x1024) and 4D (256x64x112x112).
- Same optimization profile as addition: vectorization, parallelization, and basic tiling.

## Target Hardware Considerations

**Intel Xeon E5-2680 v4 (Broadwell):**
- 28 physical cores (2 sockets x 14 cores), no hyperthreading
- AVX2 + FMA: 256-bit vectors = 4 FP64 lanes per register
- Cache: L1d 32KB/core, L2 256KB/core, shared L3 ~35MB/socket
- 2 NUMA nodes

**Key implications:**
- **Tiling** must target L1 (32KB) and L2 (256KB) working set sizes for compute-bound ops.
- **Vectorization** targets 4 FP64 lanes (AVX2). No AVX-512 available.
- **Parallelization** should target 28 threads for full utilization, with awareness of NUMA topology.
- **Register pressure** is a concern with aggressive unrolling on Broadwell (16 YMM registers).

## Intent Structure Justification

### Intent 1: Iteration Space Blocking and Data Locality (HIGH)
The most impactful optimization class for compute-bound operations (matmul, conv, pooling). Without tiling, working sets for the benchmark shapes far exceed cache capacity. For example, matmul_1024_128_1024 has operands totaling ~24MB in FP64, vastly exceeding L2 (256KB). Tiling to L1/L2-friendly blocks provides the largest single performance gain.

Loop interchange complements tiling by ensuring the innermost loops access memory with stride-1 patterns. Promotion and packing further improve cache utilization by ensuring tiled operand slices are contiguous in memory. Im2col lowering transforms convolution's complex sliding-window access pattern into a regular contraction structure that is much easier to tile and optimize.

### Intent 2: SIMD Vectorization and Instruction Throughput (HIGH)
AVX2 FMA instructions can execute 2 FLOP/cycle/lane x 4 lanes = 8 FP64 FLOP/cycle per core. Without vectorization, throughput is limited to scalar operations (1-2 FLOP/cycle). Vectorization is essential for both compute-bound (matmul, conv) and memory-bound (add, relu) operations.

Loop unrolling exposes instruction-level parallelism that keeps the FMA pipeline full. Loop peeling handles non-divisible trip counts cleanly. Canonicalization is placed here because it is critically needed before vectorization: it folds dynamic shapes from tiling/promotion into static types, enabling the vectorizer to determine correct vector widths.

### Intent 3: Thread-Level Parallelism and Work Distribution (HIGH)
With 28 physical cores, thread-level parallelism provides up to 28x speedup for perfectly parallel work. All benchmark operations have outer parallel dimensions suitable for distribution.

Two parallelization strategies are included: tiling-based (flexible granularity control) and thread-count-based (simple, direct distribution). Split reduction enables parallelization of reduction dimensions (e.g., the K dimension in matmul) when outer parallel dimensions are insufficient for full core utilization.

## Transformation Summary

| # | Transformation | Intent | Description |
|---|---------------|--------|-------------|
| 1 | Tiling | Cache Locality | Partition iteration space into cache-fitting blocks |
| 2 | Loop Interchange | Cache Locality | Reorder loops for stride-1 access |
| 3 | Promotion | Cache Locality | Copy tiled operands to contiguous buffers |
| 4 | Packing | Cache Locality | Reorganize data layout with inner blocking |
| 5 | Im2col Lowering | Cache Locality | Convert sliding-window to contraction |
| 6 | Vectorization | SIMD | Map loops to SIMD vector operations |
| 7 | Loop Unrolling | SIMD | Reduce overhead, expose ILP |
| 8 | Loop Peeling | SIMD | Handle remainders for clean vectorization |
| 9 | Canonicalization | SIMD | Normalize IR for downstream transforms |
| 10 | Tiling-Based Parallelization | Parallelism | Tile and distribute to threads |
| 11 | Thread-Count Parallelization | Parallelism | Distribute by thread count |
| 12 | Split Reduction | Parallelism | Parallelize reduction dimensions |

Total: 12 transformations across 3 intents, all HIGH priority.
