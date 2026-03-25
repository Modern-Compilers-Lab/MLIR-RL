# Layer 1 — Action Enumeration Reasoning (v11)

## Input Analysis

The input is a `linalg.matmul` operation on 2D tensors of `f64` type, wrapped in a timing harness. From a loop-nest perspective, this is a canonical 3-deep loop nest with two parallel dimensions (i, k — output rows and columns) and one reduction dimension (j — the contraction/accumulation axis). Memory access patterns include stride-1 access along one operand's innermost dimension and strided access along the other, making data layout and loop ordering critical.

## Target Hardware Considerations

- **Intel Xeon E5-2680 v4 (Broadwell)**: 28 physical cores, 2 NUMA nodes, no HT.
- **AVX2 + FMA**: 4 FP64 lanes per 256-bit vector register. No AVX-512.
- **Cache hierarchy**: L1d ~32KB, L2 ~256KB, L3 ~tens of MB shared per socket.
- The workload is compute-bound for large matrices but becomes memory-bound without proper tiling/packing.

## Optimization Intent Derivation

### Intent 1: Data Locality & Reuse Optimization (HIGH priority)

This is the single most impactful optimization category for dense loop nests on CPU. Matrix multiplication has O(N^3) compute on O(N^2) data, so temporal reuse is enormous — but only if working sets fit in cache.

- **Tiling**: Partitioning the iteration space into blocks is essential to keep working sets within L1/L2/L3 cache capacities. Multi-level tiling maps naturally to the cache hierarchy. This is the foundational transformation on which almost all other optimizations build.

- **Packing**: After tiling, the sub-matrices accessed within a tile may still have poor stride patterns (e.g., column-major access in a row-major layout). Packing copies tile data into contiguous buffers, eliminating TLB misses and cache conflict misses. This is the standard technique used in all high-performance BLAS libraries (GotoBLAS, OpenBLAS, BLIS).

### Intent 2: SIMD & Instruction-Level Performance (HIGH priority)

The Broadwell microarchitecture can issue 2 FMA operations per cycle on 256-bit vectors. Without vectorization, throughput drops by 4x (FP64) or 8x (FP32). Without unrolling, the processor cannot fill FMA pipelines due to loop overhead and data dependencies.

- **Vectorization**: Mapping an innermost loop dimension to SIMD lanes enables AVX2 FMA utilization. For FP64, this means processing 4 elements per vector instruction. The transformation restructures loop bodies so that loads, stores, and arithmetic operate on vector-width chunks.

- **Loop Unrolling**: Replicating the loop body across multiple iterations exposes independent operations to the out-of-order execution engine, enables register-level data reuse across unrolled iterations, and amortizes loop control overhead. For matmul, unrolling along the output dimensions creates a register tile that maximizes FMA throughput.

### Intent 3: Loop Structure & Parallelism (MEDIUM priority)

With 28 cores available, thread-level parallelism is important for large matrices. However, parallelization is secondary to single-core efficiency (tiling + vectorization) because over-parallelization of small tiles hurts performance. Loop interchange is a prerequisite enabler for both vectorization and parallelization.

- **Loop Interchange**: Reordering loops changes which dimension is innermost, directly affecting memory access stride patterns and vectorization legality. For a 3-deep matmul nest, the loop order determines whether the innermost access is stride-1 (vectorizable) or strided (scalar). Interchange is also needed to place parallel loops outermost for efficient thread distribution.

- **Parallelization**: Distributing outer loop iterations across the 28 available cores. For large matrices, this provides near-linear speedup on parallel dimensions. The choice of which loop to parallelize (and the grain size) affects load balance and cache behavior across NUMA nodes.

## Summary of Enumerated Actions

| # | Action | Intent | Priority |
|---|--------|--------|----------|
| 1 | Tiling | Data Locality & Reuse | HIGH |
| 2 | Packing | Data Locality & Reuse | HIGH |
| 3 | Vectorization | SIMD & ILP | HIGH |
| 4 | Loop Unrolling | SIMD & ILP | HIGH |
| 5 | Loop Interchange | Loop Structure & Parallelism | MEDIUM |
| 6 | Parallelization | Loop Structure & Parallelism | MEDIUM |

These 6 transformations form a complete, minimal action space for optimizing dense loop nests on the target CPU. They are all kernel-agnostic, parameterizable, and composable — suitable for a hierarchical RL policy.
