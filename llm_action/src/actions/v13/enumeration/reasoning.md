# Layer 1 — Optimization Reasoning: Action Enumeration v13

## Input Analysis

The RL training inputs are `linalg.matmul` operations on 2D tensors (e.g., `tensor<256x256xf64>` @ `tensor<256x512xf64>`). From a loop-nest perspective, this is a **triply-nested loop** with two parallel dimensions and one reduction dimension, exhibiting a classic contraction pattern. The iteration space is regular and dense, making it highly amenable to structured loop transformations.

## Target Hardware Considerations

- **Intel Xeon E5-2680 v4 (Broadwell)**: 28 physical cores, 2 NUMA nodes, no SMT.
- **SIMD**: AVX2 + FMA — 256-bit vectors → 4 FP64 lanes or 8 FP32 lanes. No AVX-512.
- **Cache**: L1d ~32KB, L2 ~256KB per core, shared L3 (~35MB per socket).
- **Implication**: Performance is dominated by (1) cache reuse of the working set, (2) utilization of FMA/SIMD throughput, and (3) multi-core parallelism.

## Optimization Intent Reasoning

### Intent 1: Data Locality & Cache Reuse (HIGH Priority)

This is the single most impactful category for loop-nest-dominated kernels on this hardware. A naively scheduled triply-nested contraction will suffer catastrophic cache misses on at least one operand due to the mismatch between iteration order and memory layout. The working set for even moderate matrix sizes (256x256 FP64 = 512KB per operand) exceeds L1 and L2.

**Key transformations:**

1. **Tiling**: The foundational transformation. Partitions the iteration space into blocks whose working set fits in L1/L2/L3. Multi-level tiling (macro tiles for L3/L2, micro tiles for L1/registers) is the backbone of every high-performance matmul implementation (BLIS, OpenBLAS, MKL).

2. **Loop Interchange**: Reorders loop nesting to ensure stride-1 memory access on the innermost dimension, maximizing spatial locality and cache line utilization. For contraction-like nests, the default loop order may not align with the optimal access pattern for all operands.

3. **Packing**: Copies operand data into contiguous, cache-line-aligned buffers that match the tile traversal order. Eliminates TLB thrashing and non-unit-stride accesses that persist even after tiling. This is what separates naive tiled matmul from library-quality performance.

4. **Promotion**: Materializes frequently reused sub-tensors into faster memory (stack buffers or registers). Ensures that the innermost micro-kernel operates entirely from L1 or registers rather than repeatedly fetching from L2/L3.

### Intent 2: Compute Throughput & SIMD Utilization (HIGH Priority)

Even with perfect data locality, performance is limited by how well the FMA units are fed. AVX2 provides 2 FMA units per core, each processing 4 FP64 elements/cycle. Achieving peak throughput requires vectorized inner loops, sufficient instruction-level parallelism (ILP), and clean loop boundaries.

**Key transformations:**

1. **Vectorization**: Maps the innermost loop iterations to SIMD vector lanes. For FP64 on AVX2, this means processing 4 elements per vector instruction. The inner loop dimension and its access pattern must be compatible with contiguous vector loads/stores.

2. **Unrolling**: Exposes ILP by replicating the loop body, enabling the CPU to pipeline independent FMA instructions across multiple registers. Critical for saturating both FMA ports on Broadwell. Also reduces loop overhead (branch prediction, induction variable updates).

3. **Peeling**: Splits a loop into a main body with a clean trip count (divisible by vector width or unroll factor) and a scalar remainder. Enables the main body to be vectorized and unrolled without runtime remainder checks.

4. **Padding**: Extends operand dimensions with dummy elements to guarantee divisibility by tile sizes, vector widths, or alignment requirements. Avoids complex remainder handling and enables uniform, optimized code paths throughout the iteration space.

### Intent 3: Parallelism & IR Simplification (MEDIUM Priority)

With 28 cores available, multi-threaded execution is essential for large problems, but the speedup is typically secondary to getting single-core performance right (tiling + vectorization). IR simplification (canonicalization) is a supporting transformation that cleans up the IR after other transformations, enabling better downstream optimization.

**Key transformations:**

1. **Parallelization**: Distributes independent outer loop iterations across threads/cores. For loop nests with parallel outer dimensions, this is straightforward. Must avoid oversubscription (28 cores) and consider NUMA effects for large tensors.

2. **Fusion**: Merges producer-consumer loop nests to avoid materializing intermediate tensors in memory. Reduces memory traffic and improves locality. Relevant when the matmul feeds into or is fed by element-wise operations (bias add, activation, etc.).

3. **Canonicalization**: Simplifies and normalizes the IR after other transformations (e.g., removes redundant operations, folds constants, simplifies affine expressions). Serves as a cleanup pass that improves the quality of downstream lowering and code generation.

## Design Rationale Summary

- **3 intents** covering the three pillars of loop-nest optimization: locality, compute throughput, and parallelism.
- **11 transformations total** (4 + 4 + 3), each a distinct macro RL action.
- All transformations are framed in loop-nest/iteration-space terms, not kernel-specific terminology.
- Each is parameterizable and kernel-agnostic, suitable for a hierarchical RL policy.
- Priority reflects empirical impact: locality and SIMD are essential on this hardware; parallelism is important but secondary.
