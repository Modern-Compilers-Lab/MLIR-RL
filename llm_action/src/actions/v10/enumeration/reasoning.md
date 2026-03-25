# Layer 1 — Optimization Reasoning for Matrix Multiplication Loop Nest

## Input Analysis

The input is a `linalg.matmul` operation computing C[I,K] = A[I,J] * B[J,K] on f64 tensors. From a loop-nest perspective, this is a triply-nested loop with two parallel dimensions (I, K) and one reduction dimension (J). Memory access patterns involve:
- A is accessed with stride-1 along J (inner reduction), stride-I along I (outer parallel)
- B is accessed with stride-1 along K (parallel), stride-J along J (reduction)
- C is accessed with stride-1 along K (parallel), stride-I along I (outer parallel)

## Target Hardware Considerations

Intel Xeon E5-2680 v4 (Broadwell):
- AVX2 + FMA: 4 FP64 lanes per 256-bit vector; peak throughput requires FMA utilization
- Cache hierarchy: L1d ~32KB, L2 ~256KB, L3 ~35MB shared per socket
- 28 physical cores across 2 NUMA nodes, no hyperthreading

## Optimization Strategy

### Intent 1: Data Locality & Reuse Optimization (HIGH)

This is the single most impactful optimization category for dense loop nests. Without tiling, the working set of a matrix multiplication exceeds cache capacity for any non-trivial problem size, leading to severe capacity misses. Tiling restructures the iteration space so that sub-blocks of the operand tensors fit within L1/L2 cache, maximizing temporal reuse of loaded data. Packing complements tiling by reorganizing data from its original strided layout into contiguous buffers, eliminating TLB misses and cache conflict misses that arise from power-of-two strides.

- **Tiling**: Partition the iteration space into blocks sized to fit working sets in cache. For a 3-loop nest, this means choosing tile sizes for each dimension such that the tiles of A, B, and C fit in L1 or L2. This is the foundational transformation.
- **Packing**: Copy sub-tiles into contiguous scratch buffers before computation. This converts strided accesses into unit-stride sequential reads, which is critical when the original layout has large strides (common in column-major or large-row matrices).

### Intent 2: SIMD & Instruction-Level Parallelism (HIGH)

Broadwell's AVX2 provides 4-wide FP64 SIMD with FMA. Failing to vectorize means using at most 1/4 of peak throughput. Vectorization requires an innermost loop with stride-1 access across a contiguous dimension, which may require interchange to achieve.

- **Vectorization**: Map the innermost loop to AVX2 vector instructions (4 FP64 elements per vector register). The loop chosen for vectorization must access at least one operand with unit stride.
- **Loop Interchange**: Reorder the loop nest to place a dimension with stride-1 memory access in the innermost position, enabling vectorization and improving spatial locality. For matmul, this may involve moving the K (column) dimension innermost.

### Intent 3: Coarse-Grain Parallelism (MEDIUM)

With 28 cores available, distributing work across cores is important for large problem sizes. However, the benefit is shape-dependent: small matrices may not have enough work to saturate all cores, and parallelization overhead can dominate. Loop unrolling complements parallelization by reducing loop overhead and exposing more independent operations for out-of-order execution within each core.

- **Parallelization**: Distribute iterations of an outer parallel loop across CPU cores. For matmul, the I or K dimensions are parallel and can be distributed without synchronization.
- **Loop Unrolling**: Replicate loop body iterations to reduce branch overhead and increase the number of independent instructions visible to the out-of-order execution engine, improving ILP and register utilization.

## Summary of Enumerated Actions

| # | Action | Intent | Priority |
|---|--------|--------|----------|
| 1 | Tiling | Data Locality | HIGH |
| 2 | Packing | Data Locality | HIGH |
| 3 | Vectorization | SIMD/ILP | HIGH |
| 4 | Loop Interchange | SIMD/ILP | HIGH |
| 5 | Parallelization | Coarse-Grain Parallelism | MEDIUM |
| 6 | Loop Unrolling | Coarse-Grain Parallelism | MEDIUM |
