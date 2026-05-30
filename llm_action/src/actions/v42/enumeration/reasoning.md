# Action Enumeration Reasoning — v42 (dataset_matmul, train split)

## Workload Analysis

The benchmark set consists of 187 instances of `linalg.matmul` on `f64` tensors:
- Operation: `C[I,K] = A[I,J] * B[J,K]` (tensor semantics)
- Data type: `f64` (8 bytes per element, 4 lanes per AVX2 256-bit register)
- Dimension ranges observed: I,J,K in {128, 256, 512, 768, 1024, 1536, 2048, 3072}
- Arithmetic intensity: O(I*J*K) FMA ops on O(I*J + J*K + I*K) data — compute-bound for all but the smallest shapes.

## Hardware Context

- Intel Xeon E5-2680 v4 (Broadwell), 28 physical cores (2x14), no HT
- AVX2 + FMA (no AVX-512): 4 FP64 lanes per 256-bit vector
- Cache: L1d ~32KB, L2 ~256KB per core, shared L3 ~35MB per socket
- 2 NUMA nodes

## Key Performance Bottlenecks for Matmul on This Hardware

1. **Cache capacity misses**: For matmul with dimensions > ~180 (for f64), a single row/column exceeds L1d. Without tiling, the working set of the inner loop body far exceeds L1, causing repeated main-memory fetches. This is the dominant bottleneck.

2. **Strided memory access**: The default loop order (i,j,k or i,k,j) may cause non-unit-stride access on one of the three operands. For row-major A[I,J] and B[J,K], the j-reduction loop accesses A row-wise (stride-1) and B column-wise (stride-K). Interchange and promotion address this.

3. **Under-utilization of SIMD units**: Without vectorization, only scalar FMA throughput is achieved (1/4 of peak). Vectorizing the innermost dimension with AVX2 is essential for f64 performance.

4. **Single-core execution**: With 28 cores available, failing to parallelize outer loops leaves 27 cores idle. For the larger shapes (1024+), this is a significant throughput loss.

## Intent Prioritization Rationale

### Intent 1: Cache Locality and Data Reuse — HIGH
Matmul's arithmetic intensity means that once data is in cache, many FLOPs can be extracted per byte loaded. But this only materializes if tiling keeps the working set within cache capacity. For f64 with dimensions 128-3072, untiled matmul causes catastrophic L1/L2 misses. Tiling is the single most impactful transformation. Loop interchange complements tiling by ensuring stride-1 access within tiles. Promotion further eliminates strided access by copying tiled slices into contiguous scratch buffers. Packing reorganizes the data layout at a higher level for panel-friendly access patterns.

### Intent 2: SIMD and Instruction-Level Parallelism — HIGH
Broadwell's peak f64 throughput requires saturating the FMA units via AVX2 vectors. Without vectorization, we achieve at most 1/4 of peak compute throughput. Two vectorization variants are enumerated: one with sequential tiling preprocessing (tile_using_for) and one with parallel tiling preprocessing (tile_using_forall), as these represent fundamentally different scheduling decisions (the parallel variant also distributes outer tiles across threads). Loop unrolling complements vectorization by exposing multiple independent FMA operations to the out-of-order pipeline, improving FMA unit utilization and hiding latency.

### Intent 3: Coarse-Grain Parallelism — MEDIUM
With 28 cores, parallelizing the outer loops of matmul provides near-linear speedup for large problem sizes. However, the benefit is shape-dependent: small matrices (128x128) may see overhead from thread management exceeding the parallel benefit. Two parallelization strategies are enumerated (tiling-based and thread-count-based) to give the RL agent flexibility.

## Transformation Count Summary
- Intent 1 (Cache Locality): 4 transformations — Tiling, Loop Interchange, Promotion, Packing
- Intent 2 (SIMD/ILP): 3 transformations — Vectorization (Sequential), Vectorization (Parallel), Loop Unrolling
- Intent 3 (Parallelism): 2 transformations — Parallelization (Tiling), Parallelization (Threads)

Total: 9 macro RL actions
