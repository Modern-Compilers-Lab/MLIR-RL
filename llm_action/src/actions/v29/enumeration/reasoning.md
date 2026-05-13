# Action Enumeration Reasoning — v29 (dataset_matmul)

## Input Characterization

The benchmark set consists of 187 `linalg.matmul` instances operating on `f64` tensors
with dimensions ranging from 128 to 3072. The operation computes `C[I,K] += A[I,J] * B[J,K]`
— a canonical 3-deep loop nest with two parallel iteration dimensions (I and K) and one
reduction dimension (J).

### Loop Nest Structure

```
for i in 0..I:        // parallel (rows of output)
  for k in 0..K:      // parallel (columns of output)
    for j in 0..J:    // reduction (contraction)
      C[i,k] += A[i,j] * B[j,k]
```

Memory access patterns:
- A[i,j]: stride-1 on j (innermost reduction), stride-J on i
- B[j,k]: stride-1 on k, stride-K on j
- C[i,k]: stride-1 on k, invariant on j (accumulation target)

### Working Set Analysis

At the sizes present in the dataset, even a single matrix exceeds L2 cache:
- 256×256×f64 = 512 KB (already exceeds L2 at 256 KB)
- 1024×1024×f64 = 8 MB
- 3072×3072×f64 = 72 MB

Naive execution causes severe capacity misses and poor temporal reuse. Cache-aware
blocking is essential.

### Hardware Constraints (Broadwell Xeon E5-2680 v4)

- AVX2 + FMA: 4 FP64 lanes per 256-bit vector register
- No AVX-512: vector width capped at 256 bits
- L1d: 32 KB per core, L2: 256 KB per core, L3: ~35 MB shared per socket
- 28 physical cores (2 sockets × 14 cores), no hyperthreading
- 2 NUMA nodes

## Optimization Intent Reasoning

### Intent 1: Data Locality and Reuse (HIGH priority)

This is the single most impactful optimization category for dense matrix multiplication.
The 3-loop nest exhibits O(N³) compute on O(N²) data — high arithmetic intensity when
data fits in cache, but catastrophically poor locality when it does not.

**Tiling** partitions the iteration space into blocks so that working tiles of A, B, and
C fit within L1/L2 cache. For a tile of size (ti, tj, tk), the working set is approximately
`ti*tj + tj*tk + ti*tk` elements × 8 bytes. Targeting L2 (256 KB), tile sizes around
128–256 per dimension for the outer level and 16–32 for inner levels are typical.

**Loop Interchange** reorders the loop nest to improve spatial locality. The default
iteration order may not be stride-1-optimal for all operands simultaneously. Permuting
loops so the innermost loop accesses the most frequently reused or stride-1 dimension
of the accumulation reduces cache pressure.

**Promotion** copies tiled operand slices into contiguous temporary buffers, eliminating
large-stride accesses and TLB pressure within the tile. This is especially valuable when
original operand strides are large (e.g., a 128-element tile from a 3072-column matrix
has stride-3072 rows). Promotion requires bufferization as a prerequisite step.

### Intent 2: SIMD and Instruction-Level Parallelism (HIGH priority)

The FMA units on Broadwell can sustain 4 FP64 multiply-add operations per cycle per core,
but only if the innermost loop is vectorizable and the pipeline is kept full.

**Vectorization** restructures the innermost loop to operate on 4-wide FP64 vectors
(AVX2). This typically requires that the innermost loop dimension is a multiple of the
vector width and accesses contiguous memory. Vectorization is often applied after tiling
to ensure tile sizes are SIMD-aligned.

**Unrolling** exposes more independent FMA operations per loop iteration, hiding FMA
latency (5 cycles on Broadwell) and reducing loop overhead. Unrolling the innermost
loop by factors of 2–8 is common. It also helps the register allocator keep multiple
accumulator registers live, maximizing FMA throughput.

### Intent 3: Thread-Level Parallelism (MEDIUM priority)

With 28 physical cores available, parallelizing the outer parallel loops is necessary to
exploit the full machine. Matmul's outer loops (i and k) are embarrassingly parallel.

**Parallelization (tiling-based)** distributes work by tiling the parallel dimensions
into chunks and mapping each chunk to a thread via `forall`. This is more flexible
because tile sizes can be tuned independently of core count.

**Parallelization (direct/num_threads)** directly distributes iterations across a fixed
number of threads. This is simpler but requires that dimension sizes are divisible by
thread count to avoid remainder handling issues in downstream lowering.

Both variants are included because they represent genuinely different RL action semantics:
tiling-based parallelization exposes tile_sizes as the parameter space, while direct
parallelization exposes num_threads.

## Why MEDIUM for Thread-Level Parallelism (not HIGH)

While parallelization is essential for absolute performance, the RL agent's primary
learning challenge lies in the intra-core optimization space (tiling, vectorization,
promotion). Parallelization over outer loops is relatively straightforward — the main
risk is oversubscription or NUMA-unaware distribution, not complex parameter search.
Marking it MEDIUM reflects that it is important but less of a learning challenge for
the RL policy compared to the cache and SIMD intents.

## Transformation Summary

| # | Transformation | Intent | Rationale |
|---|---------------|--------|-----------|
| 1 | Tiling | Data Locality | Cache blocking for L1/L2 reuse |
| 2 | Loop Interchange | Data Locality | Stride optimization for spatial locality |
| 3 | Promotion | Data Locality | Contiguous buffers eliminate stride/TLB issues |
| 4 | Vectorization | SIMD/ILP | Exploit AVX2 4-wide FP64 FMA units |
| 5 | Unrolling | SIMD/ILP | Pipeline fill, reduce loop overhead |
| 6 | Parallelization (Tiling) | Thread Parallelism | Distribute tiles across 28 cores |
| 7 | Parallelization (Direct) | Thread Parallelism | Direct thread mapping |
