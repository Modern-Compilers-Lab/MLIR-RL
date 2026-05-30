# Layer 1 — Action Enumeration Reasoning (v47)

## Benchmark Analysis

**Dataset**: `dataset_matmul` (train split, 187 instances)
**Operation**: `linalg.matmul` — a rank-3 loop nest computing C[I,K] += A[I,J] * B[J,K]
**Data type**: f64 (double-precision, 8 bytes per element)
**Dimension ranges**: I, J, K ∈ {128, 256, 512, 768, 1024, 1536, 2048, 3072}

### Loop Nest Structure

The `linalg.matmul` operation corresponds to a 3-deep perfectly nested loop:
- **Loop 0 (I)**: parallel — rows of the output
- **Loop 1 (K)**: parallel — columns of the output
- **Loop 2 (J)**: reduction — contraction/inner-product dimension

Two of the three loops are parallel, and one is a reduction. This structure is favorable for both parallelization and vectorization.

### Memory Access Patterns

- **A[I,J]**: accessed along J (contiguous in row-major) for each I; reused across all K values
- **B[J,K]**: accessed along K (contiguous) for each J; reused across all I values
- **C[I,K]**: accessed along K (contiguous) for each I; accumulated across J (reduction)

The key data-reuse opportunities:
- A is reused across the K dimension
- B is reused across the I dimension
- C is reused across the J (reduction) dimension

Without tiling, the working set for a single row of C computation includes an entire row of A (J elements) and the entire B matrix (J×K elements), which for typical sizes (e.g., J=1024, K=1024) is 8 MB for B alone — far exceeding L2 cache.

### Target Hardware Constraints

- **Intel Xeon E5-2680 v4 (Broadwell)**:
  - 28 physical cores (2 sockets × 14 cores), 2 NUMA nodes
  - AVX2 + FMA (no AVX-512)
  - f64 vector width: 4 elements per 256-bit register
  - L1d: 32 KB/core (~4K doubles, ~64×64 tile footprint)
  - L2: 256 KB/core (~32K doubles)
  - Shared L3 per socket: ~35 MB

### Optimization Reasoning

**1. Data Locality and Cache Reuse (HIGH priority)**

For compute-bound matmul, arithmetic intensity is O(N) per element (each output element requires N multiply-adds). However, achieving this theoretical intensity requires the data to be present in cache. Without tiling, the memory traffic scales with the full matrix size, converting a compute-bound kernel into a memory-bound one.

- **Tiling**: The single most impactful transformation. By blocking all three loop dimensions, we ensure that tiles of A, B, and C fit in L1/L2 cache. For f64 with 32KB L1, a tile of ~32×4 or 16×8 doubles per operand is feasible. For L2 (256KB), tiles of ~64×64 or similar can hold all three operand tiles.

- **Loop Interchange**: The default loop order I→K→J may not be optimal depending on the memory layout and which dimension is innermost. Permuting the loop order can improve spatial locality — for instance, making J (reduction) the innermost loop gives good stride-1 access to B when K is the middle loop, while making K innermost gives stride-1 access to C. The optimal permutation depends on the tiling and vectorization strategy.

**2. Compute Throughput via SIMD Exploitation (HIGH priority)**

The matmul inner loop performs fused multiply-add (FMA) operations that map directly to AVX2 FMA instructions. Without vectorization, only scalar execution is used, leaving 75% of the hardware FP throughput unused (4-wide SIMD).

- **Vectorization (Sequential)**: The standard vectorization approach: tile the innermost loop to match the SIMD width (4 for f64/AVX2), then vectorize. This uses sequential (for-loop) tiling as preprocessing, keeping the outer loop structure intact. The sequential tiling produces a loop with a trip count equal to the vector width, which the vectorizer can lower to SIMD operations.

- **Vectorization (Parallel)**: An alternative vectorization approach where the preprocessing tiling uses `forall` (parallel tiling) instead of sequential `for`. This combines vectorization with thread-level distribution of the outer tiles. The inner tiles are vectorized while the outer tiles can be distributed across threads.

**3. Coarse-Grain Parallelism (HIGH priority)**

With 28 physical cores and compute-bound matmul workloads, multi-core parallelism can provide up to 28× speedup. The two parallel dimensions (I and K) provide ample parallelism for most shapes in the benchmark set.

- **Parallelization (Tiling-based)**: Tile the parallel dimensions and distribute tile iterations across threads using `forall`. This approach gives control over the granularity of work per thread and naturally integrates with the tiling hierarchy (outer parallel tiles → inner sequential/vectorized tiles).

- **Parallelization (Thread-based)**: Directly map a fixed number of threads to the iteration space. Simpler but requires that thread count divides the iteration count. Useful when the iteration space is already appropriately sized.

Both approaches are included because they interact differently with subsequent transformations and have different shape-dependent effectiveness.

## Intent Structure Summary

| Intent | Priority | Transformations |
|--------|----------|----------------|
| Data Locality & Cache Reuse | HIGH | Tiling, Loop Interchange |
| SIMD Compute Throughput | HIGH | Vectorization (Sequential), Vectorization (Parallel) |
| Coarse-Grain Parallelism | HIGH | Parallelization (Tiling), Parallelization (Threads) |

Total: 3 intents, 6 transformations — forming a focused action catalog for matmul optimization on the target hardware.
