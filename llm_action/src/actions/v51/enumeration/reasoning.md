# Action Enumeration Reasoning — dataset_add (v51)

## Workload Analysis

The target operation is `linalg.add` on 4D tensors of type `f64`. This is a purely **elementwise** operation: each output element is computed as the sum of the corresponding elements from two input tensors, with no reductions, no cross-iteration data dependencies, and no data reuse.

### Memory-Bound Characterization

- **Arithmetic intensity**: O(1) — exactly 1 floating-point addition per 3 memory accesses (2 reads from inputs + 1 write to output), each 8 bytes wide (f64).
- **Implication**: Performance is entirely limited by **memory bandwidth**, not compute throughput. The CPU's FMA/ALU units will be idle most of the time waiting for data.
- **All loops are parallel**: There are no reduction dimensions. Every iteration is independent.

### Data Footprint

Tensor shapes range from small (e.g., 7x14x7x14 ≈ 55 KB across 3 tensors) to very large (e.g., 112x112x120x150 ≈ 5.1 GB across 3 tensors). The L1d cache is ~32 KB, L2 is ~256 KB, and shared L3 is ~tens of MB per socket. For the majority of shapes in this dataset, the total working set far exceeds cache capacity, making cache-efficient data traversal essential.

### Loop Structure

The `linalg.add` on a 4D tensor produces a 4-deep perfectly nested loop with no loop-carried dependencies. The default iteration order (dim0 → dim1 → dim2 → dim3) accesses memory contiguously along the last dimension (row-major / C-order), which is already favorable for spatial locality. However, without tiling, the streaming access pattern defeats cache reuse across the outer dimensions.

## Optimization Strategy

Given the memory-bound nature of elementwise add, the optimization priorities are:

1. **Cache-efficient data movement** (HIGH): Tile the iteration space so that the active working set (tiles of both inputs + output) fits within L1 or L2 cache. This converts a single long streaming pass into many small cache-resident passes, improving effective bandwidth utilization. Loop interchange can further ensure optimal access patterns after tiling.

2. **SIMD vectorization** (HIGH): AVX2 can process 4 f64 values per instruction (256-bit vectors). For a bandwidth-bound operation, wider loads/stores directly translate to higher throughput up to the memory bandwidth ceiling. Vectorization requires the innermost loop to have a trip count matching the vector width, which is achieved via preprocessing tiling.

3. **Multi-core parallelism** (MEDIUM): With 28 physical cores and all loops being embarrassingly parallel, distributing outer iterations across threads provides near-linear speedup for large tensors. However, NUMA effects and memory bandwidth saturation (the shared memory bus becomes the bottleneck) may limit scaling beyond a certain thread count. For smaller shapes, parallelization overhead may outweigh benefits.

## Transformation Selection Rationale

### Tiling
Essential for cache locality. Without tiling, large tensors stream through cache with no temporal reuse. With tiling, each tile's 3 slices (input A, input B, output C) fit in L1/L2, enabling the processor to complete all work on that tile before moving to the next.

### Loop Interchange
After tiling, the relative ordering of tiled loops determines access stride patterns. Reordering loops can ensure the innermost untiled loop accesses contiguous memory. Even for the untiled case, interchange can be beneficial if a non-default dimension ordering provides better prefetch behavior for certain shapes.

### Vectorization (two variants)
The elementwise add maps directly to SIMD vector add instructions (`vaddpd` for f64 on AVX2). Two preprocessing strategies are enumerated per the framework requirements:
- **Sequential tiling**: tiles to vector width using `for` loops — straightforward, no thread overhead.
- **Parallel tiling**: tiles to vector width using `forall` — combines vectorization with thread distribution in one step.

### Parallelization (two variants)
All loops are parallel, so coarse-grain distribution is straightforward:
- **Tiling-based**: tile outer dimensions into chunks, distribute chunks across threads.
- **Thread-count-based**: directly split iterations by thread count — simpler but requires divisibility.
