# Layer 1 — Action Enumeration Reasoning (v38)

## Benchmark: dataset_matmul (train split, 187 instances)

## Input Analysis

The input is a `linalg.matmul` operation computing C[I,K] += A[I,J] * B[J,K] in f64 precision. This corresponds to a 3-deep loop nest:
- Two outer parallel loops (over I and K dimensions of the output)
- One inner reduction loop (over J, the shared/contraction dimension)

The shapes span a wide range: I, J, K drawn from {128, 256, 512, 768, 1024, 1536, 2048, 3072}. This means working sets range from small (fitting in L2) to very large (exceeding L3), making cache-aware optimization critical across the board.

## Hardware Context

Target: Intel Xeon E5-2680 v4 (Broadwell), 28 cores (2x14), no HT.
- AVX2 + FMA: 4 FP64 lanes per 256-bit vector register.
- Cache: L1d 32KB, L2 256KB, shared L3 ~35MB per socket.
- No AVX-512.

## Performance Bottleneck Analysis

Matrix multiplication is classically compute-bound with O(I*J*K) FLOPs vs O(I*J + J*K + I*K) data. The key to high performance is:

1. **Maximizing data reuse in cache** — tiling the loop nest so that operand tiles fit in L1/L2 and are reused across the reduction dimension. Without tiling, capacity misses dominate for any non-trivial matrix size.

2. **Exploiting SIMD throughput** — the target has AVX2+FMA capable of 4 FP64 FMAs per cycle. Scalar code uses only ~25% of peak throughput. Vectorizing the innermost computation along a contiguous dimension is essential.

3. **Utilizing multiple cores** — with 28 cores available, parallelizing outer parallel loops provides linear speedup for large matrices. For the largest shapes (e.g., 3072x3072), single-core execution is impractically slow.

## Intent Derivation

### Intent 1: Cache Locality and Data Reuse (HIGH)

This is the highest-impact optimization category. For the matmul loop nest:
- **Tiling** partitions the iteration space into blocks sized to fit in L1/L2 cache, enabling temporal reuse of loaded data across iterations of the reduction loop.
- **Loop Interchange** reorders loop dimensions to ensure the innermost loop accesses memory with stride-1 (column-contiguous) pattern, maximizing spatial locality and cache line utilization.
- **Promotion** copies tiled operand slices into contiguous temporary buffers, eliminating non-unit strides and TLB pressure from large-stride access patterns in the original tensor layout.

### Intent 2: Compute Throughput Maximization (HIGH)

Equally critical for a compute-bound kernel:
- **Vectorization** maps the innermost loop iterations to SIMD instructions (4 f64 lanes with AVX2). A preprocessing tiling step shapes a loop dimension to the vector width before lowering to SIMD. Two variants are warranted:
  - Sequential variant: preprocessing tiling is purely sequential, suitable when parallelism is handled separately.
  - Parallel variant: preprocessing tiling distributes outer tiles across threads, fusing parallelism with vectorization for reduced scheduling overhead.
- **Loop Unrolling** increases instruction-level parallelism (ILP) by replicating loop bodies, allowing the CPU pipeline to overlap independent FMA operations and hide latency.

### Intent 3: Multi-Core Parallelism (MEDIUM)

Rated MEDIUM because the benefit is shape-dependent: large matrices see near-linear speedup across 28 cores, while small shapes may be dominated by thread-creation overhead.
- **Tiling-based parallelization** tiles a parallel loop dimension and distributes tiles across threads, providing natural load balancing and cache-friendly work partitioning.
- **Direct parallelization** maps loop iterations directly to threads without tiling, simpler but requires iteration counts divisible by thread count.

## Transformation Count Summary

- Intent 1 (Cache Locality): 3 transformations — Tiling, Loop Interchange, Promotion
- Intent 2 (Compute Throughput): 3 transformations — Vectorization, Parallel Vectorization, Loop Unrolling
- Intent 3 (Multi-Core Parallelism): 2 transformations — Tiling-Based Parallelization, Direct Parallelization

Total: 8 macro RL actions across 3 intents.
