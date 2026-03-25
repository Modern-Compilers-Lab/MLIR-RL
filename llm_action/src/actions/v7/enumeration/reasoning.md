# Layer 1 — Action Enumeration Reasoning (v7)

## Input Analysis

The input is a **matrix multiplication** kernel expressed as `linalg.matmul` operating on 2D tensors of f64. The template uses parameterized dimensions `[I]x[J]` times `[J]x[K]` producing `[I]x[K]`, with a concrete instance of 128x256 times 256x128.

From a loop-nest perspective, `linalg.matmul` corresponds to a **triply-nested loop** with two parallel outer loops (iterating over rows of the output and columns of the output) and one reduction loop (the contraction/accumulation dimension). The memory access pattern involves:
- A streaming read along one dimension of the left operand,
- A strided (or streaming, depending on layout) read of the right operand,
- A streaming write to the output.

## Hardware Context

Target: Intel Xeon E5-2680 v4 (Broadwell), AVX2+FMA, no AVX-512.
- FP64 vector width: 4 lanes (256-bit AVX2).
- L1d: 32KB, L2: 256KB, L3: shared ~35MB.
- 28 physical cores across 2 NUMA nodes.

## Optimization Reasoning

### Intent 1: Data Locality via Loop Tiling (HIGH priority)

Matrix multiplication is the textbook case for tiling. The triply-nested loop has O(N^3) compute on O(N^2) data, but naive execution streams through memory with poor temporal reuse. Tiling the iteration space into blocks that fit in L1/L2 cache dramatically improves data reuse. For the concrete 128x256x128 instance, the working set already partially fits in L2, but for the general template with arbitrary [I],[J],[K], tiling is essential.

Multi-level tiling (e.g., tile for L2 then tile again for L1/register) is a well-known technique for dense linear algebra on CPUs with deep cache hierarchies.

### Intent 2: SIMD Exploitation (HIGH priority)

AVX2 with FMA provides 4 FP64 FLOPs per cycle per lane (multiply-add). Without vectorization, the kernel cannot approach peak throughput. The innermost loop (or a tiled inner loop) must be mapped to SIMD lanes. This requires:
- **Vectorization**: mapping a loop dimension to vector lanes.
- Potentially **loop interchange** to ensure the vectorized dimension has unit-stride memory access.

Loop interchange is a prerequisite enabler — if the innermost loop iterates over a non-contiguous dimension, vectorization will use gather loads (slow on Broadwell). Reordering loops so the innermost dimension is contiguous enables efficient vector loads/stores.

### Intent 3: Parallelism Exploitation (MEDIUM priority)

For larger problem sizes, exploiting the 28 available cores via parallel execution of outer loop iterations is important. However, for the RL action space, parallelization is a simpler transformation — it primarily involves distributing outer parallel loops across threads. Its priority is medium because:
- For small matrices, parallelization overhead may exceed benefit.
- The primary performance bottleneck for single-core execution is data locality and vectorization.
- Still, it's a standard and important action for the general case.

## Transformation Selection

I select the following macro RL actions:

1. **Tiling** — The fundamental blocking transformation for cache locality. Parameterized by tile sizes per loop dimension. This is the highest-impact single transformation for dense linear algebra.

2. **Loop Interchange** — Reordering loop dimensions to improve memory access patterns (stride-1 access for vectorization, better spatial locality). Works on the loop band to produce a permutation.

3. **Vectorization** — Mapping a loop dimension to SIMD vector lanes. On AVX2 with FP64, this means 4-wide operations. Essential for approaching peak throughput.

4. **Parallelization** — Distributing parallel loop iterations across CPU cores. Important for scaling to the full 28-core machine on larger problem sizes.

## Grouping into Intents

- **Data Locality Optimization** (HIGH): Tiling, Loop Interchange — both directly improve cache behavior and memory access patterns.
- **Compute Throughput Optimization** (HIGH): Vectorization — directly maps computation to SIMD hardware.
- **Scalability via Parallelism** (MEDIUM): Parallelization — exploits multi-core hardware for larger workloads.
