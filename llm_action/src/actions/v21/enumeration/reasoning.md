# Action Enumeration Reasoning — v21

## Input Analysis

The input is a `linalg.matmul` operation in MLIR, representing a dense matrix multiplication `(I x J) @ (J x K) -> (I x K)` on `f64` data. This is a canonical triply-nested loop nest with two parallel dimensions (I, K) and one reduction dimension (J).

From a loop-nest perspective, the operation has:
- Three tightly nested loops iterating over a 3D iteration space.
- Three tensor operands with distinct access patterns (row-major, column-major-like, and output).
- A reduction (contraction) across the shared inner dimension.
- Arithmetic intensity of O(n^3) compute over O(n^2) data, making it compute-bound for large sizes.

## Hardware Context

Target: Intel Xeon E5-2680 v4 (Broadwell), 28 physical cores, AVX2+FMA, no AVX-512.
- Vector width: 4 lanes for f64 (256-bit AVX2).
- Cache: L1d ~32KB, L2 ~256KB per core, shared L3 per socket.
- No hyper-threading. Two NUMA nodes.

## Intent Selection Rationale

### Intent 1: Data Locality & Cache Reuse (HIGH)

This is the single most impactful optimization category for matrix multiplication and dense loop nests in general. Without tiling, the working set far exceeds cache capacity for non-trivial sizes, causing massive DRAM traffic. The three transformations selected address different aspects:

- **Tiling**: Partitions the iteration space so that the working set of each tile fits in L1 or L2 cache. This is the foundational transformation — nearly all high-performance matmul implementations are built on multi-level tiling.
- **Loop Interchange**: Reorders loops within a tile to maximize spatial locality (stride-1 access on the innermost loop) and to expose reduction loops at the right nesting depth for accumulation. For a 3-loop nest, the loop order determines whether memory accesses sweep contiguously or with large strides.
- **Promotion**: After tiling, operand sub-tensors may still have non-unit strides in memory (e.g., a column slice of a row-major matrix). Promotion copies these sub-tensors into contiguous local buffers, eliminating stride penalties and enabling aligned, predictable vector loads. This requires bufferization as a prerequisite step within the action.

### Intent 2: Compute Throughput & SIMD Utilization (HIGH)

Once data locality is addressed, the bottleneck shifts to maximizing arithmetic throughput. AVX2+FMA can perform 4 fused multiply-add operations per cycle on f64. The transformations here target this:

- **Vectorization**: Maps the innermost loop dimension onto SIMD vector lanes. For f64 with AVX2, this means processing 4 elements per vector instruction. This is essential to approach peak FLOP/s.
- **Unrolling**: Replicates loop body iterations to increase instruction-level parallelism (ILP), reduce loop overhead (branch mispredictions, counter increments), and enable the CPU's out-of-order engine to overlap independent FMA chains. Unroll-and-jam (unrolling an outer loop and fusing the copies into the inner loop body) is particularly effective for register-level data reuse.
- **Packing**: Rearranges the data layout of tensor operands so that the data accessed by the innermost (vectorized) loop is contiguous in memory and properly aligned. This transforms irregular access patterns into sequential vector loads, complementing vectorization.

### Intent 3: Work Partitioning & Parallelism (MEDIUM)

With 28 physical cores available, distributing work across threads is necessary for large problems. However, this is rated MEDIUM because:
- For small to moderate matrix sizes (as in the 128x256x128 example), parallelization overhead may dominate.
- The primary performance wins come from cache and SIMD optimizations first.
- Parallelization is shape-dependent — it helps most when the outer loop iteration count is large enough to partition meaningfully.

- **Parallelization**: Distributes outer parallel loop iterations across CPU threads. The outer (non-reduction) loops of a matmul are embarrassingly parallel and map naturally to thread-level distribution.
- **Peeling**: Separates partial/remainder iterations (boundary tiles) from the main loop body. This allows the main loop to operate on full, uniform tiles — enabling clean vectorization and parallelization without conditional checks — while a separate epilogue handles the leftover iterations.

## Design Decisions

1. **No compound actions**: Tiling, interchange, and vectorization are kept as separate macro actions even though they are commonly applied together. This gives the RL policy maximum flexibility in choosing the order and parameters.

2. **Promotion separate from Packing**: Though related, promotion operates at the buffer level (copying tiled sub-tensors into contiguous local memory) while packing operates at the data layout level (rearranging tensor element ordering). They target different abstraction levels and can be applied independently.

3. **Peeling included under parallelism**: Peeling is a supporting transformation that enables clean parallelization and vectorization by isolating boundary cases. It is not a standalone optimization but an enabler for other transformations.

4. **No fusion actions**: The input is a single operation with no producer-consumer relationships to fuse. Fusion would be relevant for multi-operation graphs but is not applicable here.
