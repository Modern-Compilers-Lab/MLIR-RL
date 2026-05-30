# Action Enumeration Reasoning — v46 (dataset_matmul)

## 1. Loop-Nest Analysis

The `linalg.matmul` operation computes C[i,k] += A[i,j] * B[j,k] and maps to a 3-deep loop nest:

- **Loop i** (parallel): iterates over rows of output C, range [0, I)
- **Loop k** (parallel): iterates over columns of output C, range [0, K)
- **Loop j** (reduction): iterates over the contraction/inner-product dimension, range [0, J)

All three operands are rank-2 tensors of f64 (8 bytes per element):
- A: I x J, B: J x K, C: I x K

The benchmark set contains 187 instances with dimensions ranging from 128 to 3072. The naming convention is `matmul_I_J_K`.

## 2. Hardware Characteristics & Implications

Target: Intel Xeon E5-2680 v4 (Broadwell), 28 physical cores (2 sockets x 14), no HT.

### Memory Hierarchy
- L1d: ~32 KB per core → fits ~4096 f64 values → e.g., a 64x64 tile of one operand
- L2: ~256 KB per core → fits ~32768 f64 values → can hold several small tiles
- L3: shared per socket, tens of MB → can hold larger working sets
- NUMA: 2 sockets, so large allocations may span NUMA domains

Working set sizes for representative shapes (f64):
- matmul_1024_1024_128: A=8MB, B=1MB, C=1MB → total ~10MB, far exceeds L1/L2
- matmul_128_128_128: A=128KB, B=128KB, C=128KB → total ~384KB, fits in L3 but not L2
- matmul_768_3072_512: A=18MB, B=12MB, C=3MB → total ~33MB, requires multi-level tiling

Conclusion: **Cache-aware tiling is essential** for virtually all shapes in this dataset.

### SIMD Capabilities
- AVX2 + FMA: 256-bit vectors → **4 f64 elements per vector register**
- 2 FMA units per core → peak 8 f64 FLOPs/cycle/core
- No AVX-512 (do not assume 512-bit widths)

Conclusion: **Vectorization targeting 4-wide f64 SIMD** is critical to approach peak throughput.

### Parallelism
- 28 physical cores across 2 NUMA nodes
- Matmul has two parallel loop dimensions (i and k), offering natural parallelism
- The reduction dimension (j) cannot be trivially parallelized without atomic updates or reduction trees

Conclusion: **Coarse-grain parallelism over outer parallel dimensions** is essential for large shapes.

## 3. Optimization Intent Selection

### Intent 1: Data Locality & Cache Efficiency (HIGH)

**Why**: Matmul has O(n^3) compute with O(n^2) data, giving a high compute-to-data ratio — but only if data can be reused from cache. Without tiling, the naive loop order causes repeated cache evictions as matrices far exceed L1/L2. For f64 on this hardware, even moderate shapes (256x256) produce working sets exceeding L2. Multi-level tiling, loop reordering for stride-1 access, and data packing to eliminate cache conflict misses are the most impactful optimizations.

**Transformations**:
- **Tiling**: The foundational transformation. Blocks the 3D iteration space so that sub-tiles of A, B, and C fit in L1/L2. The RL agent selects tile sizes per dimension.
- **Loop Interchange**: Reordering loops changes memory access patterns. For matmul, the default loop order may not produce stride-1 access for all operands; interchange enables the innermost loop to traverse the fastest-varying dimension.
- **Packing**: After tiling, sub-tiles of operands may still have non-unit strides in memory (especially B in row-major layout). Packing copies tile data into contiguous buffers, eliminating TLB misses and cache line conflicts.
- **Promotion**: Copies tiled operand data into contiguous local buffers at the memref level. This requires bufferization as a preprocessing step. Promotion targets the outer tile level so copy cost is amortized over many inner iterations.

### Intent 2: SIMD & Compute Throughput (HIGH)

**Why**: The matmul inner loop is a multiply-accumulate — a perfect match for FMA instructions. AVX2 provides 4-wide f64 FMA, and Broadwell has 2 FMA units per core for a peak of 8 FLOPs/cycle. Without vectorization, the code runs at scalar throughput (1 FLOP/cycle), leaving ~87.5% of compute capability unused. Loop unrolling further helps by exposing independent FMA operations to fill both FMA pipelines and hide latency.

**Transformations**:
- **Vectorization (Sequential)**: SIMD-lower the innermost loop(s) after preprocessing with sequential tiling (tile_using_for) to match vector widths. The sequential variant keeps the outer tile loop as a regular for-loop.
- **Vectorization (Parallel)**: SIMD-lower the innermost loop(s) after preprocessing with parallel tiling (tile_using_forall), which distributes outer tiles across threads. Combines vectorization with work distribution.
- **Loop Unrolling**: Unroll an inner loop by a fixed factor to reduce loop overhead, expose instruction-level parallelism across multiple FMA units, and improve register utilization. Particularly effective after tiling when trip counts are known and small.

### Intent 3: Multi-Core Parallelism & Buffer Management (HIGH)

**Why**: With 28 cores and no hyperthreading, distributing the parallel dimensions of matmul across cores provides up to 28x potential speedup on large shapes. The two parallel dimensions (i and k) can be tiled and distributed. Additionally, when parallelism is combined with tiling, each thread's working set should ideally be in local NUMA memory — promotion of tiled operands to local buffers supports this. Both tiling-based and thread-count-based parallelization strategies are valuable: tiling-based gives more control over granularity, while thread-based provides simpler mapping to hardware threads.

**Transformations**:
- **Parallelization (Tiling-based)**: Partition the iteration space of parallel loop dimensions via tiling, then distribute the outer tile loops across threads. Reduction dimensions are automatically excluded. Tile sizes control granularity — larger tiles reduce scheduling overhead, smaller tiles improve load balance.
- **Parallelization (Thread-based)**: Directly distribute parallel loop iterations across a fixed number of threads. Simpler parameterization (just num_threads) but requires the iteration count to be divisible by the thread count for correct subsequent lowering.

## 4. Design Decisions

- **All 3 intents are HIGH priority**: For matmul on this hardware, cache tiling, SIMD vectorization, and multi-core parallelism are all essential — omitting any one leaves major performance on the table.
- **Two vectorization variants**: The prompt explicitly requires enumerating both sequential and parallel preprocessing variants as separate actions.
- **Two parallelization variants**: Following the guidance to provide both tiling-based and thread-count-based parallelization as separate transformations.
- **Promotion under Intent 3**: Promotion creates local buffer copies of tiled data, which is critical when threads operate on different tiles — placing it under parallelism and buffer management reflects its role in per-thread data locality.
- **9 total transformations** (4 + 3 + 2): Each maps directly to a discrete RL macro action with clear parameterization.

Wait — Intent 3 has only 2 transformations, below the 3-minimum. Promotion is moved to Intent 3 to satisfy the constraint and because its buffer-management nature aligns with ensuring efficient per-thread data access in a parallel context.

**Final count**: Intent 1 (3 transformations: Tiling, Loop Interchange, Packing) + Intent 2 (3 transformations: Vectorization Sequential, Vectorization Parallel, Loop Unrolling) + Intent 3 (3 transformations: Parallelization Tiling, Parallelization Threads, Promotion) = 9 total.
