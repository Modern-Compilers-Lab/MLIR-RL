# Action Enumeration Reasoning — Pooling NCHW Max (v34)

## Operation Analysis

The target operation is `linalg.pooling_nchw_max`, a max-pooling reduction over a sliding window on NCHW-layout tensors.

### Loop-Nest Structure

`linalg.pooling_nchw_max` decomposes into a 6-deep loop nest:

- **4 parallel outer loops**: N (batch), C (channels), OH (output height), OW (output width)
- **2 reduction loops**: KH (kernel height), KW (kernel width)

The body computes: `output[n,c,oh,ow] = max(output[n,c,oh,ow], input[n, c, oh*stride_h + kh*dilation_h, ow*stride_w + kw*dilation_w])`

### Memory Access Characteristics

- **Output tensor**: written contiguously along OW (innermost parallel dimension), good spatial locality.
- **Input tensor**: read with stride > 1 in the spatial dimensions due to pooling stride (commonly stride=2). This means consecutive output elements read non-adjacent input elements, leading to poor cache line utilization unless tiled.
- **Kernel/window tensor**: very small (1x1, 3x3, or 7x7), trivially fits in registers.
- **Overall**: the operation is **memory-bound**. The max comparison is a single FP64 comparison — negligible compute relative to data movement cost.

### Shape Diversity in the Dataset (250 instances)

- Batch sizes: 128, 256
- Channel counts: 96 to 512
- Input spatial dimensions: 14x14 to 240x240
- Kernel sizes: 1x1 (strided subsample), 3x3, 7x7
- Output spatial dimensions: 4x4 to 120x120
- Derived strides: typically 1 or 2

Key observation: input tensors can be very large (e.g., 128x128x112x112 FP64 = ~1.6GB). Working sets vastly exceed cache sizes, making tiling essential.

### Baseline Performance Gap

MLIR baseline execution is approximately 8-10x slower than PyTorch JIT across the benchmark set. This confirms that the default lowering without optimization transformations is far from optimal and significant speedups are achievable through standard loop-nest optimizations.

## Intent Selection Rationale

### Intent 1: Cache Locality and Data Reuse — HIGH Priority

This is the highest-impact intent because pooling is memory-bound. The input tensors are often very large and the strided access pattern from pooling strides leads to poor cache utilization. Tiling the iteration space to fit working sets in L1/L2, combined with loop reordering for stride-1 access within tiles, directly addresses the primary bottleneck. Promotion of tiled input slices into contiguous buffers further eliminates the strided access penalty.

**Transformations chosen:**
1. **Tiling** — partitions the large iteration space into cache-friendly blocks. Essential for any large tensor operation on this hardware (L1=32KB, L2=256KB per core).
2. **Loop Interchange** — reorders loops within tiles for better spatial locality. The default loop order may not be optimal for the NCHW layout with strided access.
3. **Promotion** — copies tiled input slices into contiguous temporary buffers, converting strided pooling reads into dense sequential reads. Particularly valuable when pooling stride > 1.

### Intent 2: SIMD Exploitation — HIGH Priority

AVX2 provides 4 FP64 lanes per vector register. The max operation maps directly to SIMD max instructions (`_mm256_max_pd`). Vectorizing the innermost parallel loop (output width) processes 4 output elements simultaneously. Since the per-element compute is trivial (one comparison), maximizing SIMD throughput is critical to keep the execution units fed. Unrolling complements vectorization by reducing loop overhead, especially important when reduction windows are tiny (1x1 or 3x3, meaning 1-9 iterations).

**Transformations chosen:**
1. **Vectorization** — maps parallel loops to AVX2 vector lanes for 4-wide FP64 max operations.
2. **Unrolling** — reduces loop overhead and exposes instruction-level parallelism. Crucial when reduction loops have very low trip counts (1-9 iterations).

### Intent 3: Coarse-Grain Parallelism — MEDIUM Priority

With batch sizes of 128-256 and channel counts up to 512, the outer parallel dimensions provide ample work to distribute across the 28 physical cores. However, this is medium priority because: (a) single-core efficiency through cache optimization and SIMD typically yields higher returns for memory-bound operations, and (b) parallelization adds overhead that may not pay off for smaller spatial dimensions. Two parallelization approaches are enumerated (tiling-based and num_threads-based) as recommended by the guidelines.

**Transformations chosen:**
1. **Parallelization (tiling-based)** — tiles outer parallel dimensions and distributes tiles across threads. More flexible, handles non-divisible iteration counts gracefully.
2. **Parallelization (num_threads-based)** — directly partitions iterations of an outer parallel loop across a fixed thread count. Simpler but requires divisibility.
