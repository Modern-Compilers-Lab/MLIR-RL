# Action Enumeration Reasoning

## Benchmark Analysis

The benchmark set consists of 1135 instances across 5 operation families targeting an Intel Xeon E5-2680 v4 (Broadwell) CPU with 28 physical cores (2 sockets x 14 cores, no HT), AVX2+FMA (256-bit, 4 FP64 lanes), and a 3-level cache hierarchy (L1d ~32KB, L2 ~256KB per core, shared L3 per socket).

### Operation Characteristics

**Compute-intensive operations (high arithmetic intensity):**
- **matmul** (186 instances): 3-deep loop nest with 2 parallel and 1 reduction dimension. Shapes range from 128 to 3072 per dimension. Working sets far exceed L2 capacity for most instances. O(I*J*K) FLOPs over O(I*J + J*K + I*K) data yields high arithmetic intensity that grows with problem size.
- **conv_2d_nchw_fchw** (277 instances): 7-deep loop nest with 4 parallel loops (N, F, OH, OW) and 3 reduction loops (C, KH, KW). Batch sizes 128-256, channels up to 512, spatial dimensions 7-56, kernel sizes 1x1 to 7x7, variable strides (1 or 2). Extremely large working sets and complex strided access patterns across multiple tensor dimensions.

**Memory-bound operations (low arithmetic intensity):**
- **add** (270 instances): 4-deep loop nest, all parallel. Pure elementwise addition: 2 loads + 1 store per element with 1 FLOP. Arithmetic intensity ~1/24 for FP64 (1 FLOP / 24 bytes transferred). Performance entirely limited by memory bandwidth.
- **relu** (148 instances): 2-4 deep loop nest, all parallel. Elementwise max(0, x): 1 load + 1 store per element with a comparison and select. Similar or lower arithmetic intensity than add.
- **pooling_nchw_max** (249 instances): 6-deep loop nest with 4 parallel loops (N, C, OH, OW) and 2 window reduction loops (KH, KW). Max-reduction over a sliding window. Arithmetic intensity depends on window size — larger windows (e.g., 7x7) approach moderate intensity, while 1x1 windows are effectively memory-bound.

### Key Optimization Considerations

1. **Working set analysis**: A matmul_512_512_512 has ~6MB working set (3 * 512^2 * 8 bytes), far exceeding L2 (256KB). Conv2d instances with batch=128 have even larger footprints. Tiling to fit cache is essential.

2. **Vectorization opportunity**: AVX2 FMA provides 4 FP64 lanes. Without SIMD utilization, 75% of peak FP64 throughput is wasted. All operations have innermost dimensions amenable to vectorization.

3. **Parallelism opportunity**: 28 cores available. All operations have embarrassingly parallel outer dimensions (batch, channels, spatial). Benchmark batch sizes (128, 256) and channel counts (32-512) provide ample parallelism.

4. **Structural complexity**: Conv2d's 7-deep nested loop structure with interleaved parallel and reduction dimensions benefits from iteration-space lowering (im2col) to expose simpler contraction patterns amenable to vectorization.

## Intent Selection Rationale

### Why 3 intents?

Three fundamental hardware bottlenecks on the Broadwell Xeon map directly to three HIGH-priority intents:
1. **Cache hierarchy utilization** — cache misses dominate execution time for compute-bound ops
2. **SIMD lane utilization** — scalar code uses only 1 of 4 FP64 vector lanes
3. **Core utilization** — single-threaded code uses only 1 of 28 available cores

### Intent 1: Data Locality and Cache Optimization (HIGH)

For matmul and conv2d (463 of 1135 instances, 41% of the set), cache locality is the most impactful single optimization category. A naive traversal of a 512x512x512 matmul re-fetches each element of the reduction-dimension operand ~512 times from main memory. Tiling to L2-sized blocks reduces this to ~1 fetch per element per tile.

Three transformations are included:
- **Tiling** — controls the working set size (foundational)
- **Loop Interchange** — ensures stride-1 access within tiles
- **Promotion** — copies tiled slices into contiguous local buffers, eliminating strided access in the micro-kernel

For memory-bound operations (add, relu), tiling structures access for cache-line efficiency and prefetcher utilization even though reuse is minimal.

### Intent 2: SIMD Exploitation (HIGH)

AVX2+FMA on Broadwell provides 4 FP64 FLOPS/cycle/lane * 4 lanes = 16 FLOPS/cycle. Without vectorization, peak throughput drops to 4 FLOPS/cycle (scalar FMA). This 4x gap affects every operation family.

Three transformations:
- **Vectorization with Sequential Tiling** — tile innermost loops to vector width using sequential for-loops, then lower to SIMD. Appropriate when parallelism is managed separately at outer levels.
- **Vectorization with Parallel Tiling** — tile innermost loops to vector width using parallel forall-loops, combining SIMD exposure with thread distribution. Exploits both parallelism levels simultaneously.
- **Im2col Lowering** — placed under SIMD because its primary effect is converting conv2d's 7-deep loop nest (which resists clean vectorization due to complex window-sliding index arithmetic) into a 3-loop contraction whose innermost reduction loop maps directly to vector FMA operations. The restructuring is fundamentally motivated by enabling effective SIMD exploitation for convolution.

Two vectorization variants are required per the system specification: one with sequential tiling preprocessing, one with parallel tiling preprocessing.

### Intent 3: Thread-Level Parallelism (HIGH)

28 cores provide up to 28x theoretical speedup for parallel outer loops. All operation families have outer parallel dimensions, and benchmark batch sizes (128, 256) offer ample parallelism.

Two transformations:
- **Parallelization via Tiling** — tile outer parallel loops and distribute via forall. Decouples tile granularity from thread count.
- **Parallelization via Thread Count** — directly partition iterations across N threads. Simpler, maps cleanly when loop bounds are divisible.

Note: Parallel Vectorization (Intent 2) also contributes to thread-level parallelism by distributing outer tiles across threads during the vectorization preprocessing step. The two dedicated parallelization actions here provide pure thread distribution without vectorization coupling, giving the RL agent flexibility to separate concerns.

Two parallelization variants are recommended per the system specification.

## Transformation Coverage Summary

Total: 8 transformations across 3 intents (3 + 3 + 2).

| Transformation | matmul | conv2d | pooling | add | relu |
|---|---|---|---|---|---|
| Tiling | HIGH | HIGH | MED | MED | MED |
| Loop Interchange | HIGH | HIGH | MED | LOW | LOW |
| Promotion | HIGH | HIGH | LOW | - | - |
| Seq. Vectorization | HIGH | HIGH | HIGH | HIGH | HIGH |
| Par. Vectorization | HIGH | HIGH | HIGH | HIGH | HIGH |
| Im2col Lowering | - | HIGH | - | - | - |
| Tiling Parallelization | HIGH | HIGH | HIGH | HIGH | HIGH |
| Thread Parallelization | HIGH | HIGH | HIGH | HIGH | HIGH |

7 of 8 transformations are universal (applicable to all 5 families). 1 is conv2d-specific (Im2col). This provides a focused action space where almost every action applies to every operation, with one specialized action for the largest operation family (conv2d, 277 instances).
