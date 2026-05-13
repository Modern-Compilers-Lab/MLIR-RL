# Action Enumeration Reasoning — v27

## Benchmark Analysis

**Benchmark set**: `paper_matmul` (train split, 4 instances)
**Operation**: `linalg.matmul` — a triply-nested loop with two parallel dimensions (I, K) and one reduction dimension (J).

### Shapes (I×J×K):
| Instance               | I   | J    | K    | FLOPs (2·I·J·K) | A size (B) | B size (B) | C size (B) |
|------------------------|-----|------|------|------------------|------------|------------|------------|
| matmul_256_256_128     | 256 | 256  | 128  | ~16.8M           | 512 KB     | 256 KB     | 256 KB     |
| matmul_256_256_512     | 256 | 256  | 512  | ~67.1M           | 512 KB     | 1 MB       | 1 MB       |
| matmul_256_512_1024    | 256 | 512  | 1024 | ~268M            | 1 MB       | 4 MB       | 2 MB       |
| matmul_256_1536_1000   | 256 | 1536 | 1000 | ~786M            | 3 MB       | 12 MB      | 2 MB       |

(All f64, so element size = 8 bytes)

### Key Observations

1. **Compute-bound workload**: Matmul has O(N³) FLOPs over O(N²) data. The arithmetic intensity (FLOPs/byte) grows with problem size, making this compute-bound for all but the smallest shapes.

2. **Data sizes exceed L1/L2 for most instances**: L1d is 32KB, L2 is 256KB per core. Even the smallest instance (matmul_256_256_128) has operands totaling ~1MB — well beyond L2. Tiling is essential to keep working sets cache-resident.

3. **f64 with AVX2**: 4 lanes per vector (256-bit / 64-bit). Peak throughput requires saturating FMA pipelines with 4-wide vector operations, demanding proper vectorization and register blocking.

4. **Two parallel dimensions (I, K)**: Both can be tiled and parallelized. The reduction dimension (J) requires accumulation and is a natural candidate for the innermost loop (to enable register reuse of the accumulator).

5. **Moderate parallelism potential**: I=256 for all instances provides up to 256 independent rows — sufficient for 28-core distribution, but the absolute problem sizes are moderate, so parallelization overhead matters.

## Intent Selection Rationale

### Intent 1: Data Locality and Cache Efficiency (HIGH)

This is the most impactful optimization category for dense matmul. The fundamental idea: restructure the iteration space so that data accessed by inner loops fits in L1/L2 cache, maximizing temporal and spatial reuse.

- **Tiling** is the foundational transformation. Multi-dimensional blocking of the I×J×K space to create tiles whose operand slices fit in cache. For a 3-loop reduction nest, tiling all three dimensions enables "register-tile at L1, cache-tile at L2" strategies.
- **Loop Interchange** controls which dimension is innermost, directly affecting memory access strides. For matmul C[i,k] += A[i,j]·B[j,k], the default linalg order may not be optimal; placing K innermost gives unit-stride writes to C and unit-stride reads from B (row-major layout), which is ideal for vectorization.
- **Promotion** copies tiled operand slices into contiguous scratch buffers, eliminating large-stride access patterns that cause TLB thrashing and cache conflict misses within tile computations.

### Intent 2: Compute Throughput Maximization (HIGH)

After tiling establishes cache-friendly blocks, the inner micro-kernel must be optimized for raw compute throughput on the FMA units.

- **Vectorization** maps the innermost loop to 4-wide f64 AVX2 operations. This is a prerequisite for approaching peak FLOP rates — without SIMD, throughput is limited to scalar FMA (1/4 of peak).
- **Unrolling** exposes multiple independent FMA operations to the out-of-order execution engine, enabling pipelining and register-level data reuse (register blocking). For matmul, unrolling the accumulation register tile (e.g., a small M×N block of C) across both a parallel and the reduction dimension is the standard approach for high-performance micro-kernels.

### Intent 3: Coarse-Grain Parallelism (MEDIUM)

The target machine has 28 cores across 2 NUMA nodes. Multi-threading the outer parallel loops can yield significant speedup, but with caveats:

- For the smallest shapes (256×256×128, ~16.8M FLOPs), thread overhead may dominate.
- For larger shapes (256×1536×1000, ~786M FLOPs), parallel speedup is substantial.
- Both tiling-based and direct parallelization approaches are included, as the prompt guidelines recommend two distinct parallelization actions.

Medium priority because: (a) it depends on tiling being done first, (b) the shapes are moderate, and (c) the primary bottleneck is cache efficiency and SIMD utilization for these problem sizes.

## Transformation Summary

| # | Transformation               | Intent                     | Rationale (brief)                                    |
|---|------------------------------|----------------------------|------------------------------------------------------|
| 1 | Tiling                       | Data Locality              | Block iteration space for cache-resident tiles       |
| 2 | Loop Interchange             | Data Locality              | Optimize memory access strides                       |
| 3 | Promotion                    | Data Locality              | Contiguous scratch buffers for tiled operands         |
| 4 | Vectorization                | Compute Throughput         | Map innermost loops to 4-wide f64 AVX2 FMA           |
| 5 | Unrolling                    | Compute Throughput         | Register blocking, pipeline utilization              |
| 6 | Parallelization (Tiling)     | Coarse-Grain Parallelism   | Tile-based thread distribution                       |
| 7 | Parallelization (Direct)     | Coarse-Grain Parallelism   | Direct iteration distribution across threads         |
