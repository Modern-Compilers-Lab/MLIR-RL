# Action Enumeration Reasoning — v54 (dataset_matmul, train split)

## Workload Analysis

The input is a single `linalg.matmul` operation on 2D f64 tensors:
- C[I,K] += A[I,J] * B[J,K]
- Three loop dimensions: two parallel (I, K indexing output rows and columns) and one reduction (J, the contraction dimension).
- 187 training instances with dimensions drawn from {128, 256, 512, 768, 1024, 1536, 2048, 3072}.
- Data type is f64 (8 bytes per element).

The operation is compute-bound with O(I*J*K) FMAs and O(I*J + J*K + I*K) memory traffic. Data reuse is high: each element of A is reused K times, each element of B is reused I times, and each element of C is accumulated over J iterations.

## Target Hardware Characteristics

- Intel Xeon E5-2680 v4 (Broadwell): 28 physical cores, 2 NUMA nodes.
- AVX2 + FMA: 256-bit vectors → 4 f64 lanes per vector register.
- No AVX-512.
- Cache: L1d ~32KB/core, L2 ~256KB/core, shared L3 ~35MB/socket.
- No hyper-threading.

## Optimization Reasoning

### Intent 1: Cache Locality and Data Reuse (HIGH)

Matrix multiplication's performance is dominated by memory hierarchy utilization. Without tiling, the working set of even a 256x256 f64 matrix (512KB) exceeds L2 capacity. For larger shapes (1024+), the full matrices span tens of MB, far exceeding even L3.

**Tiling** partitions the iteration space into blocks sized to fit in L1/L2 cache, maximizing temporal reuse of loaded data. For matmul, a well-chosen tile of the 3D iteration space ensures that the tile of A, tile of B, and tile of C all reside in cache simultaneously. This is the single most impactful transformation for matmul performance.

**Loop Interchange** reorders loop dimensions to ensure the innermost loop iterates over contiguous memory (column-major vs. row-major access patterns). For matmul, the default loop order may not align with the optimal access pattern for vectorization and cache line utilization. Interchange after tiling controls which dimension becomes the innermost tight loop, directly affecting spatial locality and vectorization friendliness.

**Promotion** copies tiled operand slices into contiguous temporary buffers, eliminating non-unit stride access patterns that arise when tiles are subviews of larger matrices. This is especially beneficial after tiling when tile rows are non-contiguous in the original matrix layout. Promotion converts strided access into dense sequential access, improving cache line utilization, enabling prefetching, and reducing TLB pressure. It requires bufferization as an internal preprocessing step (tile → bufferize → promote → canonicalize).

### Intent 2: SIMD Vectorization (HIGH)

On Broadwell with AVX2+FMA, each core can execute 2 FMA operations per cycle on 256-bit vectors (4 f64 elements). Unlocking this throughput requires:
- The innermost loop to operate on contiguous data with trip count matching the vector width.
- Tiling to create a tight inner loop of the right size for SIMD lowering.

There are two distinct vectorization strategies based on how the preprocessing tiling is performed:

**Sequential Vectorization** tiles the target loops using `tile_using_for` to create inner loops with trip counts matching vector sizes, then vectorizes those inner loops. The outer tiled loops remain sequential. This is the standard single-threaded vectorization path and is always applicable.

**Parallel Vectorization** tiles using `tile_using_forall`, which simultaneously distributes the outer tiles across threads and creates inner loops sized for vectorization. This combines thread-level parallelism with SIMD in a single action. It is particularly effective for large matmuls where both parallelism and vectorization are needed.

### Intent 3: Thread-Level Parallelism (MEDIUM)

The target machine has 28 cores. For large matmul shapes (e.g., 2048x2048x2048), a single core would be severely underutilized. Distributing work across cores via parallel outer loops provides near-linear speedup on parallel dimensions.

This is ranked MEDIUM rather than HIGH because: (a) small shapes (128x128x128) may not benefit from parallelization overhead, (b) the benefit is shape-dependent (parallelization helps most when the parallel dimension count far exceeds the core count), and (c) the parallel vectorization action in Intent 2 already provides one path to multi-threading.

**Tiling-Based Parallelization** tiles an outer parallel dimension and distributes the tiles across threads. The tile size controls granularity and load balance.

**Thread-Count-Based Parallelization** directly partitions the iteration space among a fixed number of threads. Requires the iteration count to be divisible by the thread count for clean partitioning.

## Summary of Actions

| # | Action | Intent | Notes |
|---|--------|--------|-------|
| 1 | Tiling | Cache Locality | Multi-level blocking for L1/L2 fit |
| 2 | Loop Interchange | Cache Locality | Reorder loops for stride-1 access |
| 3 | Promotion | Cache Locality | Copy tiles to contiguous buffers |
| 4 | Sequential Vectorization | SIMD | tile_using_for + vectorize |
| 5 | Parallel Vectorization | SIMD | tile_using_forall + vectorize |
| 6 | Tiling-Based Parallelization | Parallelism | Distribute tiled outer loops |
| 7 | Thread-Count Parallelization | Parallelism | Direct thread partitioning |

Total: 3 intents, 7 transformations (3 + 2 + 2).
