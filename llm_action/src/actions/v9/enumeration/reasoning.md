# Layer 1 — Action Enumeration Reasoning (v9)

## Input Analysis

The input is a matrix multiplication kernel expressed as `linalg.matmul` operating on 2D tensors of `f64` type. The kernel computes C = A × B where A is [I×J], B is [J×K], and C is [I×K]. A concrete instance uses shapes like 128×256 @ 256×128 → 128×128.

From a loop-nest perspective, this is a triply-nested loop with two parallel outer dimensions (I, K) and one reduction dimension (J). Memory access patterns are: A is accessed row-major along J (inner), B is accessed column-major along J (inner), and C is accumulated in-place.

## Target Hardware Considerations

The target is Intel Xeon E5-2680 v4 (Broadwell):
- AVX2 + FMA: 4 FP64 lanes per 256-bit vector register
- L1d ~32KB, L2 ~256KB per core, shared L3
- 28 physical cores across 2 NUMA nodes
- No AVX-512

For FP64 matmul, the key performance factors are:
1. **Cache reuse**: The working set of the three matrices easily exceeds cache capacity at typical problem sizes. Tiling is essential to keep working tiles in L1/L2.
2. **SIMD utilization**: AVX2 provides 4-wide FP64 FMA. The innermost loop must be vectorizable with contiguous memory access.
3. **Multi-core parallelism**: With 28 cores available, outer parallel loops should be distributed to exploit all cores.

## Intent Prioritization Rationale

### Intent 1: Cache Locality and Data Reuse (HIGH)

Matrix multiplication has O(N³) compute with O(N²) data, giving high arithmetic intensity—but only if data is reused from cache. Without tiling, every element of B is streamed from memory for each row of A, leading to catastrophic cache miss rates. Tiling the iteration space to fit working sets in L1/L2 is the single most impactful optimization.

Packing (promotion to contiguous buffers) complements tiling by eliminating stride-related cache conflicts and ensuring the tiled data lies in contiguous memory, which improves prefetch efficiency and avoids TLB misses. On Broadwell with 256KB L2, packing tiled blocks is a well-known technique from high-performance BLAS implementations (GotoBLAS/OpenBLAS pattern).

### Intent 2: SIMD Exploitation (HIGH)

AVX2 FMA can perform 4 FP64 multiply-accumulate operations per cycle per core. To saturate this, the innermost loop must operate on contiguous data aligned to vector width. This requires:
- Vectorization of the innermost computation loop to use SIMD instructions.
- Loop interchange to ensure the dimension being vectorized corresponds to contiguous memory access (unit-stride). Without proper loop ordering, vectorization either fails or produces gather/scatter operations that negate performance gains.

Both transformations are essential and complementary: interchange enables vectorization by creating the right memory access pattern.

### Intent 3: Coarse-Grain Parallelism (MEDIUM)

The outer loops of a tiled matmul are embarrassingly parallel. With 28 cores available, distributing these across threads can yield near-linear speedup. This is rated MEDIUM rather than HIGH because:
- Single-core performance improvements from tiling + vectorization typically dominate.
- Parallelization is relatively straightforward once tiling creates independent work chunks.
- Over-parallelization with small problem sizes can hurt due to thread overhead.

## Transformation Selection

The 5 transformations selected cover the fundamental optimization knobs for dense loop nests on CPU:

1. **Tiling** — the foundational transformation for cache locality in loop nests.
2. **Packing** — data layout transformation that ensures tiled data is contiguous in memory.
3. **Vectorization** — maps computation to SIMD lanes for throughput.
4. **Loop Interchange** — reorders loops to enable contiguous memory access patterns needed by vectorization and cache line utilization.
5. **Parallelization** — distributes independent loop iterations across CPU cores.

These are all standard, composable, and broadly applicable to any dense loop-nest workload. They map directly to well-known RL action primitives and can be parameterized independently.
