# Action Enumeration Reasoning — v23

## Input Analysis

The RL system input is a single `linalg.matmul` operation on 2D tensors of `f64` type, wrapped in a timing harness. The operation represents a dense matrix multiplication `C = A * B` where `A: [I×J]`, `B: [J×K]`, `C: [I×K]`. The concrete instance uses `128×256 @ 256×128 → 128×128`.

From a loop-nest perspective, `linalg.matmul` lowers to a triply-nested loop with two parallel dimensions (I, K) and one reduction dimension (J), performing `C[i,k] += A[i,j] * B[j,k]`.

## Hardware Constraints (Intel Xeon E5-2680 v4 Broadwell)

- **SIMD**: AVX2 + FMA — 4 FP64 lanes per 256-bit vector. No AVX-512.
- **Cores**: 28 physical (2×14), no SMT. 2 NUMA nodes.
- **Cache**: L1d 32KB/core, L2 256KB/core, shared L3 ~35MB/socket.
- **Key bottleneck**: For matmul, arithmetic intensity is O(N) (compute O(N³) over O(N²) data), so the kernel is compute-bound for moderate/large N but memory-bandwidth-sensitive at tile boundaries and for cold data. Cache tiling is essential to keep data in L1/L2.

## Intent Derivation

### Why two primary intents?

HPC optimization of dense loop nests decomposes naturally into two concerns:

1. **Memory hierarchy optimization**: Make data access patterns cache-friendly. For matmul, naive row-major access of B has stride-J in the innermost loop, causing cache thrashing. Tiling creates small blocks that fit in L1/L2. Loop interchange can improve stride patterns. Packing rearranges data for contiguous access. Promotion materializes tiles into local buffers.

2. **Compute throughput maximization**: Once data is cache-resident, maximize arithmetic throughput. Vectorization exploits AVX2 FMA for 4 FP64 ops/cycle. Unrolling exposes ILP to fill the FMA pipeline (Broadwell can sustain 2 FMA ops/cycle with sufficient ILP). Loop peeling handles non-vector-multiple remainders. Parallelization distributes work across 28 cores.

These two intents are both HIGH priority because:
- Without cache tiling, even vectorized code stalls on memory latency.
- Without vectorization, cache-friendly code wastes 75% of potential throughput (scalar vs 4-wide SIMD).
- Parallelization is essential for large matrices but is secondary because single-core efficiency must be established first.

## Transformation Selection

### Intent 1 — Cache Locality and Memory Hierarchy Optimization

| Transformation | Rationale |
|---|---|
| **Tiling** | Foundational. Partitions the 3D iteration space (I,J,K) into blocks sized for L1/L2. For 128×128 FP64 tiles: 128KB, fits L2. Smaller tiles (e.g., 32×32: 8KB) fit L1. Without tiling, B columns are accessed with stride J, causing capacity misses. |
| **Loop Interchange** | Reorders loop dimensions to place the reduction dimension or the dimension with best stride in the innermost position. For matmul with row-major A and B, the optimal inner loop accesses contiguous elements. |
| **Packing** | After tiling, sub-matrices of A and B may still have non-unit stride in memory. Packing copies and rearranges tile data into a contiguous buffer with a layout optimized for the inner kernel, eliminating TLB misses and enabling predictable prefetching. |
| **Promotion** | Copies tiled operand slices into contiguous local (stack-allocated) buffers. This eliminates aliasing concerns and enables the compiler to generate more aggressive loads/stores. Operates at memref level, requiring internal bufferization. |

### Intent 2 — Compute Throughput and Parallelism

| Transformation | Rationale |
|---|---|
| **Vectorization** | Maps the innermost loop to AVX2 SIMD operations. With FP64, each vector instruction processes 4 elements. For FMA: `C[i,k:k+4] += A[i,j] * B[j,k:k+4]`. This is the single most impactful compute transformation. |
| **Loop Unrolling** | Replicates the loop body N times to reduce branch overhead and expose instruction-level parallelism. Broadwell's out-of-order engine benefits from 2-4× unrolling to keep both FMA ports busy. |
| **Loop Peeling** | Separates remainder iterations (when trip count is not a multiple of vector width or tile size) into a separate loop. This allows the main loop body to assume clean, aligned iteration counts, enabling efficient vectorization without masking overhead. |
| **Parallelization** | Distributes iterations of parallel (non-reduction) loops across threads. For matmul, the I and K dimensions are embarrassingly parallel. With 28 cores, distributing the outer loop provides coarse-grain parallelism. Tile sizes for distribution should account for NUMA locality. |

## Design Decisions

- **No fusion/fission**: The input is a single operation; there are no adjacent loops to fuse or complex bodies to distribute.
- **No im2col**: The input is matmul, not convolution; im2col lowering is not applicable.
- **No canonicalization as a separate action**: Canonicalization is an infrastructure pass typically embedded within other actions (e.g., post-promotion canonicalization) rather than a standalone RL decision.
- **Parallelization under "Compute Throughput"**: Thread-level parallelism is grouped with compute throughput because all transformations in this intent aim to maximize utilization of available compute resources (SIMD units, pipelines, and cores).
- **2 intents rather than 3+**: With 8 transformations for a single matmul operation, a 2-intent structure provides the cleanest separation of concerns. A 3rd intent for parallelism alone would be under-populated.
