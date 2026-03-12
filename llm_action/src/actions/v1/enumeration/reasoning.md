# Layer 1 — Action Enumeration Reasoning (v1)

## Analysis of Input Operations

The RL training inputs consist of three kernel families, all expressed as structured MLIR `linalg` operations on tensors:

1. **Matrix Multiplication** (`linalg.matmul`): A 3-deep loop nest (I, J, K) with two parallel dimensions (I, K) and one reduction dimension (J). Memory access patterns include stride-1 access on the innermost dimension of one operand but strided access on the other, creating a classic cache-locality challenge.

2. **2D Convolution** (`linalg.conv_2d_nchw_fchw`): A 7-deep loop nest (N, F, C, OH, OW, KH, KW) with 4 parallel and 3 reduction dimensions. This has high reuse potential but complex multi-dimensional access patterns with small kernel windows.

3. **Generic Element-wise** (`linalg.generic` with all-parallel iterators): A 5-deep loop nest with all parallel dimensions and identity indexing maps. Memory-bound with simple access patterns; performance is dominated by memory bandwidth and vectorization efficiency.

## Target Hardware Considerations

The Intel Xeon E5-2680 v4 (Broadwell) defines key optimization priorities:
- **L1d 32KB / L2 256KB / shared L3**: Tiling must fit working sets into L1 or L2 for reuse-heavy kernels (matmul, conv). For element-wise ops, tiling improves spatial locality and TLB behavior.
- **AVX2 with FMA (256-bit)**: FP64 gives 4 lanes per vector. Vectorizing the innermost loop is critical for all kernel types.
- **28 physical cores, 2 NUMA nodes**: Outer-loop parallelism is available and important for large tensors, but oversubscription must be avoided.
- **No AVX-512**: Vector widths are capped at 256-bit; do not assume 512-bit operations.

## Optimization Intent Selection

### Intent 1: Data Locality Optimization (HIGH priority)
For matmul and convolution, the dominant performance bottleneck on this hardware is cache misses. Tiling restructures loop nests to keep working sets in L1/L2 cache. Loop interchange reorders dimensions to improve stride patterns (e.g., ensuring stride-1 access on innermost loops). These two transformations are the most impactful for reuse-heavy kernels and also benefit element-wise operations through improved spatial locality.

### Intent 2: SIMD Exploitation (HIGH priority)
AVX2+FMA provides 4 FP64 lanes. Vectorization maps the innermost loop dimension onto SIMD lanes. Unrolling (and unroll-and-jam) exposes independent operations for ILP and helps the backend fill vector pipelines. These are essential for all three kernel types — without vectorization, performance is severely limited on this hardware.

### Intent 3: Coarse-Grain Parallelism (MEDIUM priority)
With 28 cores across 2 NUMA nodes, distributing outer parallel loops across threads is important for large tensors. However, the benefit is shape-dependent: small tensors may not have enough work to distribute, and parallelization overhead can dominate. This makes it MEDIUM priority — beneficial but not universally essential.

## Transformation Selection Rationale

- **Tiling**: The single most important transformation for cache locality. One action with a tile_sizes vector covers all kernel types and loop depths. Zero in a position means "don't tile that dimension."
- **Loop Interchange**: Reorders loop dimensions to improve stride patterns. Critical for matmul (ensuring reduction loop position), beneficial for convolution (reordering spatial/channel loops). Works on any loop nest.
- **Vectorization**: Maps a loop to SIMD lanes. Essential for AVX2 utilization. A single action with target loop and vector width parameters.
- **Unrolling**: Exposes ILP, reduces loop overhead, enables register-level reuse. Unroll-and-jam variant provides cross-iteration reuse. Parameterized by unroll factor.
- **Loop Fusion**: Merges adjacent loop nests sharing iteration space to reduce memory traffic (producer-consumer patterns). More relevant when operations are sequenced. Keeps intermediate data in registers/cache.
- **Parallelization**: Distributes parallel loop iterations across threads. Important for large tensors on 28-core system. Parameterized by which loop level to parallelize.

## Why Not Other Transformations

- **Packing / Layout Transformation**: Important in practice but involves complex memory management and is more of a Layer-2 concern about how tiling is materialized. Could be added in future versions.
- **Peeling**: Primarily a cleanup transformation for tile remainders — better handled as part of tiling implementation in Layer 2.
- **Kernel-specific lowering (im2col)**: While useful for convolution, it changes the algorithm rather than restructuring loop nests. Could be considered in future versions but risks over-specialization at Layer 1.
