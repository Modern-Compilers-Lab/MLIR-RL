# Layer 1 — Action Enumeration Reasoning (v8)

## Input Analysis

The input is a `linalg.matmul` operation computing C[I,K] = A[I,J] * B[J,K] on f64 tensors. This is a classic triply-nested loop (over I, J, K dimensions) with:
- Two parallel loops (I and K — the output dimensions),
- One reduction loop (J — the contraction/accumulation dimension),
- Regular, strided memory access patterns across all three operands.

## Target Hardware Considerations

Intel Xeon E5-2680 v4 (Broadwell):
- AVX2 + FMA: 4 f64 lanes per 256-bit vector register.
- L1d 32KB, L2 256KB per core, shared L3 ~35MB.
- 28 physical cores across 2 NUMA nodes, no hyperthreading.
- Register file: 16 YMM registers — register pressure is a real concern for f64 with deep unrolling.

## Optimization Reasoning

### Intent 1: Data Locality Optimization (HIGH priority)

Matrix multiplication is memory-bandwidth-limited for large matrices. The working set of a naive triply-nested loop far exceeds cache capacity. Two key transformations address this:

**Tiling**: Partitions the iteration space into blocks so that sub-tiles of A, B, and C fit within L1/L2 cache. This is the single most impactful transformation for matmul on CPUs. For f64, an L1-friendly tile might be ~32x32 elements (32*32*8B = 8KB per matrix tile, three tiles ~24KB fits in 32KB L1d). L2-friendly tiles can be larger.

**Packing (Data Layout Transformation)**: After tiling, the sub-tiles of A and B may still have poor stride patterns (e.g., accessing a column of a row-major matrix). Packing copies sub-tiles into contiguous, cache-line-aligned buffers. This eliminates TLB misses, conflict misses, and enables the microkernel to stream data efficiently. Packing is what separates naive tiled matmul from BLAS-level performance.

### Intent 2: Instruction-Level Parallelism and SIMD Exploitation (HIGH priority)

Once data locality is ensured via tiling, the innermost computation must be mapped to AVX2 FMA instructions efficiently.

**Vectorization**: The innermost loop should be lowered to use 256-bit vector operations (4 f64 elements per vector). This requires that the vectorized dimension has unit stride in at least one operand. Vectorization provides up to 4x throughput improvement for f64.

**Loop Unrolling**: Unrolling the loop(s) adjacent to the vectorized loop increases the number of independent FMA operations in flight, hiding FMA latency (5 cycles on Broadwell). With 16 YMM registers, a well-unrolled microkernel can maintain ~12 accumulator registers while streaming A and B tiles. However, over-unrolling causes register spills, so the factor must be bounded.

### Intent 3: Coarse-Grain Parallelism (MEDIUM priority)

With 28 cores available, parallelizing outer loops is important for large matrices. However, this is secondary to getting single-core performance right (tiling + vectorization).

**Parallelization**: Distributes iterations of outer parallel loops (I and/or K dimensions) across threads. The choice of which loop to parallelize and the granularity (chunk size) matters for load balance and NUMA effects.

**Loop Interchange**: Reordering loops can improve spatial locality by ensuring the innermost loop iterates along the stride-1 dimension of the most-accessed operand. For matmul, the canonical loop order may not be optimal depending on the storage layout; interchange enables selecting the best order before vectorization.

## Summary

The three intents cover the classic optimization hierarchy for dense linear algebra on CPUs:
1. **Data locality** (tiling + packing) — reduces memory traffic by orders of magnitude.
2. **SIMD + ILP** (vectorization + unrolling) — exploits hardware execution units.
3. **Parallelism** (parallel distribution + loop interchange) — scales across cores.

This ordering reflects priority: a well-tiled, vectorized single-core matmul will outperform a parallelized but poorly-tiled one.
