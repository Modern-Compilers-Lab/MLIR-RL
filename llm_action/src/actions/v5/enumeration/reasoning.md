# Layer 1 — Action Enumeration Reasoning (v5)

## Input Analysis

The input is a `linalg.matmul` operation on 2D tensors of type `f64`, computing C = A × B where A is [I×J], B is [J×K], and C is [I×K]. From a loop-nest perspective, this is a triply-nested loop with two parallel dimensions (I, K) and one reduction dimension (J). The memory access pattern involves:
- A is accessed row-major along I (stride-1 along J in the inner dimension),
- B is accessed column-major along J (stride-K along J),
- C is accessed row-major along I (stride-1 along K).

The key performance bottleneck for naive execution is poor spatial locality on B (strided access along the reduction dimension) and the large working set that exceeds cache capacity for non-trivial sizes.

## Hardware Considerations

Target: Intel Xeon E5-2680 v4 (Broadwell)
- AVX2 + FMA: 4 lanes for f64 (256-bit vectors)
- L1d: 32KB, L2: 256KB, L3: ~35MB shared per socket
- 28 physical cores across 2 NUMA nodes
- No AVX-512

For f64 matmul, the key performance drivers are:
1. **Tiling** to fit working sets into L1/L2 cache
2. **Packing** to eliminate strided access on B and ensure contiguous micro-panels
3. **Vectorization** at AVX2 width (4 × f64) along the appropriate loop
4. **Parallelization** of outer loops across cores

## Intent Derivation

### Intent 1: Data Locality and Cache Utilization (HIGH)
This is the most critical optimization for matrix multiplication on CPU. Without tiling, the working set for even moderate matrix sizes far exceeds L1/L2 cache. Tiling partitions the iteration space into blocks whose data footprint fits in cache, maximizing temporal reuse. Loop interchange complements tiling by reordering loops to improve spatial locality (stride-1 access patterns). Packing goes further by copying sub-matrices into contiguous buffers, eliminating TLB misses and enabling predictable stride-1 access on all operands.

### Intent 2: SIMD Vectorization Exploitation (HIGH)
AVX2 provides 4-wide f64 FMA operations, which are essential to approach peak FLOPS. Vectorization maps the innermost computation to SIMD instructions. Unrolling along one or more dimensions exposes more independent FMA operations, improving instruction-level parallelism and keeping the FMA pipeline saturated. These two transformations together are necessary to extract near-peak throughput from the micro-kernel.

### Intent 3: Coarse-Grain Parallelism (MEDIUM)
With 28 cores available, parallelizing outer tiled loops across cores is important for large problems. Fusion of producer-consumer loop nests (when applicable in larger computational graphs) reduces intermediate materialization and synchronization. Priority is MEDIUM because single-core optimization (tiling + vectorization) typically dominates the speedup for the matmul kernel itself, and parallelization is relatively straightforward once tiling is in place.

### Intent 4: Loop Nest Regularization (MEDIUM)
Real-world matrix dimensions are not always multiples of tile sizes or vector widths. Peeling separates remainder iterations from the main loop body, enabling the main body to assume clean trip counts for vectorization and tiling. Canonicalization simplifies the IR (folding constants, removing redundant operations) to enable downstream passes to pattern-match and apply transformations more reliably. These are enabling transformations that improve the effectiveness of all other optimizations.
