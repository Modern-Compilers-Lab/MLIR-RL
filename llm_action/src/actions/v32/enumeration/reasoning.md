# Action Enumeration Reasoning — v32

## Benchmark Input

Operation family: `linalg.matmul` (dataset: `dataset_matmul`, 187 instances, train split).

Loop-nest view:
```
for i in [0, I):
  for k in [0, K):
    for j in [0, J):
      C[i,k] += A[i,j] * B[j,k]
```
Shapes range from 128 to 2048 per dimension (e.g., `matmul_1024_1024_128`, `matmul_2048_128_128`, `matmul_512_512_512`).

## Hardware Context

- Intel Xeon E5-2680 v4 (Broadwell), 28 physical cores, 2 NUMA sockets
- AVX2 + FMA; **no AVX-512**
- FP64 vector width: 4 lanes (256-bit)
- L1d: 32 KB/core, L2: 256 KB/core, L3: shared per socket (~35 MB)
- No SMT

---

## Intent Selection Rationale

### Intent 1 — Cache Locality (HIGH)

For large matrix shapes (e.g., 1024×1024×512), the default triple-nested loop touches:
- A: I×J elements, B: J×K elements, C: I×K elements

Without tiling, even a single row of A (J=1024 FP64 = 8 KB) fits in L2, but the full
matrix B (1024×128×8B = 1 MB) does not, causing repeated L3 or DRAM fetches per
output row. The solution is multi-level tiling.

**Transformations selected:**

- **Tiling**: The primary cache-blocking transformation. Partitions the I, J, K
  dimensions into tiles sized to fit in L1/L2. Enabling data reuse across all three
  operands. Universally applicable, highest ROI.

- **Loop Interchange**: After tiling, loop order matters for access strides. Moving the
  reduction loop (J) innermost over the output dimension (K) can flip the access pattern
  on B from column-strided to row-strided (unit stride), which is critical for cache-line
  utilization. Also useful pre-vectorization.

- **Promotion**: Once a tile is defined, copying the tile of A, B, or C into a compact
  contiguous local buffer eliminates any remaining non-unit strides or indirect indexing
  inside the inner tile. This is especially important for B, which is accessed along J
  (reduction dimension) and may have a non-unit stride depending on layout. Requires
  bufferization internally (memref level). Targets the outer tile level for amortized copy cost.

### Intent 2 — SIMD Vectorization (HIGH)

AVX2 provides 256-bit SIMD with dual-issue FMA. For FP64, this is 4 lanes.
Peak throughput: 4 lanes × 2 FMA/cycle × 2 FLOP/FMA = 16 FLOP/cycle per core.
Without vectorization, effective throughput is ~4× lower.

**Transformations selected:**

- **Vectorization**: The direct mechanism to map inner loop iterations to SIMD lanes.
  After tiling and potentially interchange, the innermost loop (typically over K or a
  packed inner dimension) should be vectorizable. This transformation performs the
  scalar-to-vector lowering. Critical for throughput.

- **Packing**: Vectorization requires unit-stride access. For B, the natural layout
  (row-major J×K) is fine for K-innermost access, but for A the access along J with
  K-innermost compute is strided. Packing reorganizes operand data into a layout that
  ensures unit-stride vector loads along the vectorized dimension. Operates as a
  data-layout transformation before the inner tile computation.

  Note: Promotion and Packing are complementary but distinct — Promotion copies a tile
  into a local buffer (cache behavior), Packing reorganizes the data order within that
  buffer (SIMD stride behavior). Layer 2 will handle their interaction.

### Intent 3 — Coarse-Grain Parallelism (HIGH)

The I and K output dimensions of matmul are embarrassingly parallel (no loop-carried
dependence). Distributing these across 28 cores provides near-linear speedup for large
matrices.

Two distinct dispatch strategies are enumerated per the system guidance:

- **Parallelization** (num_threads-based): Directly annotates a parallel loop for
  thread dispatch with a fixed thread count. Straightforward for shapes where the
  iteration count divides evenly by the thread count. Works at the IR level by
  converting the loop to a parallel construct.

- **Parallel Tiling** (tiling-based distribution): Applies a coarse outer tiling first,
  then dispatches the outer tile loop in parallel. This is more flexible (tile size
  controls per-thread working set), naturally composable with inner cache tiling, and
  provides NUMA locality control by ensuring each thread processes a contiguous chunk.
  Conceptually distinct from cache-locality Tiling because the block size targets
  thread granularity, not cache fitting, and the outer loop is explicitly parallelized.

---

## Output Summary

| Intent | Priority | Transformations |
|---|---|---|
| Cache Locality | HIGH | Tiling, Loop Interchange, Promotion |
| SIMD Vectorization | HIGH | Vectorization, Packing |
| Coarse-Grain Parallelism | HIGH | Parallelization, Parallel Tiling |

All three intents are rated HIGH because each addresses a distinct, non-overlapping
performance bottleneck (memory bandwidth, compute throughput, core utilization) and
all three are essential on this Broadwell target to approach peak performance for
matmul-shaped loop nests.
