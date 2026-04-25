# Layer 1 — Action Enumeration Reasoning (v12)

## Input Analysis

The input is a **linalg.matmul** operation on tensors of type `f64`, computing `C[I,K] = A[I,J] * B[J,K]`. This is a classic 3-deep nested loop with:
- Two parallel dimensions (I, K) and one reduction dimension (J).
- Regular, dense memory access patterns.
- Compute intensity of O(I*J*K) multiply-accumulate operations over O(I*J + J*K + I*K) memory accesses.

## Target Hardware Recap

Intel Xeon E5-2680 v4 (Broadwell):
- 28 physical cores (2x14), 2 NUMA nodes, no HT.
- AVX2 + FMA: 256-bit vectors → 4 FP64 lanes.
- L1d ~32KB, L2 ~256KB, shared L3 ~35MB per socket.
- No AVX-512.

## Optimization Reasoning

### Intent 1: Data Locality & Cache Reuse (HIGH Priority)

Matrix multiplication's performance is dominated by memory hierarchy utilization. For a naive 3-loop nest, one of the input matrices is accessed with non-unit stride, and working sets easily exceed cache capacity. The most impactful optimization family is **tiling** to create cache-resident blocks, combined with **loop interchange** to ensure innermost accesses are stride-1, and **promotion/packing** to materialize tiles in contiguous scratchpad buffers that eliminate TLB and cache-line conflicts.

- **Tiling**: Partitions the iteration space into blocks that fit in L1/L2 cache. This is the single most important transformation for matmul-class loop nests. Multi-level tiling (L2 then L1) is standard practice.
- **Loop Interchange**: Reorders loop dimensions to place the loop with stride-1 access innermost, critical for spatial locality and SIMD friendliness.
- **Promotion**: Allocates temporary buffers for tile-sized slices, converting potentially strided accesses into contiguous reads. Reduces cache conflict misses.
- **Packing**: Transforms the data layout of promoted tiles to ensure sequential memory access, particularly important when the original tensor has large strides between accessed elements.

### Intent 2: SIMD & Instruction-Level Parallelism (HIGH Priority)

AVX2+FMA provides 4 FP64 FMA operations per cycle per core. Maximizing vector utilization requires the innermost computation to be mappable to SIMD lanes, with sufficient independent operations to fill the FMA pipeline (5-cycle latency on Broadwell → need ~20 independent FP64 FMAs in flight per core).

- **Vectorization**: Maps the innermost loop dimension to SIMD vector lanes (4-wide for FP64 on AVX2). This is the primary mechanism for extracting data-level parallelism.
- **Loop Unrolling**: Expands loop bodies to expose multiple independent operations, enabling better instruction scheduling, register reuse, and hiding FMA latency through software pipelining.
- **Peeling**: Separates remainder iterations (when trip count is not a multiple of vector width or unroll factor) into a separate loop, allowing the main loop to be fully vectorized without masking overhead.

### Intent 3: Thread-Level Parallelism (MEDIUM Priority)

With 28 cores across 2 NUMA nodes, coarse-grained parallelism over outer loops is important for large problem sizes. However, parallelization must be balanced against per-thread working set size and NUMA effects.

- **Parallelization**: Distributes iterations of outer parallel loops across threads. For matmul, the I and K dimensions are embarrassingly parallel; J (reduction) requires care.
- **Fusion**: Merges producer-consumer loop nests to reduce intermediate materialization and improve temporal locality. Relevant when matmul feeds into subsequent operations or when multi-level tiling creates opportunities for fusing tile-level computation with data movement.
- **Loop Distribution**: Splits a loop nest into independent loop nests to enable different optimization strategies per partition, or to isolate parallel from reduction dimensions.

## Design Decisions

1. **No kernel-specific actions**: All transformations are framed in generic loop-nest terms, applicable beyond matmul.
2. **No compound actions**: Each transformation is atomic (e.g., "Tiling" and "Fusion" are separate, not "Tile and Fuse").
3. **No parameter binding**: No specific tile sizes, unroll factors, or loop indices are prescribed — that is Layer 2's responsibility.
4. **Priority assignment**: Cache locality and SIMD are HIGH because they dominate single-core performance on this hardware. Parallelism is MEDIUM because it requires larger problem sizes to amortize overhead. IR preparation is MEDIUM because it enables other transformations but doesn't directly improve performance.
