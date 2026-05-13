# Layer 1 — Action Enumeration Reasoning (v22)

## Input Analysis

The input is a `linalg.matmul` operation — a canonical 3-deep loop nest computing C[i,k] += A[i,j] * B[j,k] over iteration space (I, J, K) where J is the reduction dimension and I, K are parallel dimensions. The concrete instance uses f64 elements with shapes 128x256 @ 256x128 -> 128x128.

From a loop-nest perspective, this is a triply-nested loop with:
- Two outer parallel dimensions (I=128, K=128)
- One inner reduction dimension (J=256)
- Three operand tensors with distinct access patterns: A[i,j] (row-major stride along j), B[j,k] (row-major stride along k), C[i,k] (output, row-major stride along k)

## Hardware Context

Target: Intel Xeon E5-2680 v4 (Broadwell), 28 physical cores (2x14, 2 NUMA nodes), AVX2+FMA (no AVX-512).

Key parameters for f64:
- Vector width: 4 lanes (256-bit AVX2)
- L1d: 32KB per core (~4K f64 values)
- L2: 256KB per core (~32K f64 values)
- L3: shared per socket (~tens of MB)
- FMA throughput: 2 FMA ops/cycle/core (theoretical peak)

## Optimization Strategy Reasoning

### Intent 1: Data Locality and Cache Efficiency (HIGH)

This is the highest-impact optimization class. A naive matmul with 128x256x128 dimensions has:
- A: 128x256 = 32K elements (256KB in f64) — fits in L2 but not L1
- B: 256x128 = 32K elements (256KB in f64) — same
- C: 128x128 = 16K elements (128KB in f64) — fits in L2

Without tiling, the innermost loop streams through entire rows/columns, causing cache thrashing. With L1-aware tiling (e.g., tiles of ~32-64 elements), working sets fit in L1d, dramatically improving reuse.

**Tiling**: The foundational transformation. Multi-level tiling (L2 tiles containing L1 tiles) is the standard approach for dense linear algebra. Tile sizes are the primary tunable parameters.

**Loop Interchange**: The reduction dimension (J) vs. parallel dimensions (I, K) ordering affects whether A or B has stride-1 access. The classic ikj or jki orderings each favor different operand access patterns. Interchange interacts critically with vectorization choices.

**Promotion**: After tiling, operand sub-blocks may still have non-unit stride access (e.g., columns of a row-major matrix). Promoting these sub-blocks into contiguous local buffers eliminates stride issues and TLB pressure. This is the standard "copy optimization" used in BLAS libraries (GOTOBLAS, OpenBLAS, BLIS all use promotion/packing).

### Intent 2: SIMD Exploitation and Compute Throughput (HIGH)

AVX2 with FMA provides 4 f64 FMA operations per vector instruction. Without vectorization, we use only 25% of peak throughput.

**Vectorization**: Maps the innermost loop dimension to SIMD lanes. For f64 with AVX2, the natural vector width is 4. The choice of which dimension to vectorize interacts with data layout — typically the fastest-varying dimension of the output is vectorized.

**Loop Unrolling**: After vectorization, unrolling the next-inner loop exposes multiple independent FMA chains to the out-of-order execution engine. This is critical for hiding FMA latency (typically 4-5 cycles) and saturating the FPU pipeline. An unroll factor of 4-8 is typical.

**Packing**: Distinct from promotion. Packing reorganizes data into a blocked/panel layout specifically designed for vector access. For example, packing B into panels of width 4 (matching the f64 vector width) ensures every vector load is contiguous and aligned, regardless of the original matrix layout. This is the "Goto's algorithm" approach used in all high-performance GEMM implementations.

### Intent 3: Work Distribution and Thread-Level Parallelism (MEDIUM)

Rated MEDIUM because:
- The concrete instance (128x128x256) is relatively small — parallelism overhead may dominate
- For larger instances, parallelism is essential but the optimal strategy depends heavily on problem size
- Parallelism is orthogonal to the single-core optimizations above and is typically applied last

**Parallelization**: The two parallel dimensions (I, K) can be distributed across cores. Tiling-based parallelization (tile then distribute tiles) provides better cache behavior than cyclic distribution because each thread works on a contiguous block.

**Loop Peeling**: When tile sizes, vector widths, or thread counts don't evenly divide loop bounds, remainder iterations prevent clean optimization. Peeling separates these remainders so the main loop body can be fully optimized. This is a necessary cleanup transformation that enables other optimizations to work on clean, divisible bounds.

## Design Decisions

1. **Three intents chosen** to reflect the three orthogonal performance axes: memory hierarchy, compute throughput, and parallelism.

2. **Promotion and Packing are separated** because they serve different purposes:
   - Promotion: contiguous copy for cache locality (placed under data locality intent)
   - Packing: layout reorganization for vector access (placed under SIMD intent)

3. **Loop Peeling included** under parallelism because its primary role in this context is enabling clean division of work across tiles/threads/vector widths. It's a support transformation that unblocks other optimizations.

4. **No fusion actions** because the input is a single operation — there's no producer-consumer pair to fuse. Fusion would be relevant for multi-operation payloads.

5. **No canonicalization/simplification as explicit action** — these are mechanical cleanup passes, not strategic optimization decisions for the RL agent.
