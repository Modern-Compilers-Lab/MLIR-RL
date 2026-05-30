# Action Enumeration Reasoning — ReLU (dataset_relu, v52)

## Workload Analysis

The target operation is **ReLU** expressed as a `linalg.generic` with identity affine maps on both input and output tensors. The body performs an element-wise maximum with zero: `cmpf ugt` followed by `arith.select`.

### Key Structural Properties

1. **Purely element-wise / embarrassingly parallel**: All iterator types are `"parallel"`. Every output element depends solely on the corresponding input element — there are no reductions, no cross-element dependencies.

2. **Memory-bandwidth bound**: The per-element computation is trivially cheap (one compare, one select). Performance is dominated by memory throughput — how fast data can be read from and written to memory.

3. **Varying rank and tensor sizes**:
   - 2D instances: e.g., `relu_128_1024` (128×1024 = 131K elements × 8B = ~1 MB)
   - 4D instances: e.g., `relu_256_96_112_112` (256×96×112×112 ≈ 308M elements × 8B ≈ 2.5 GB)
   - Range spans from tensors fitting in L2 cache to tensors far exceeding total LLC capacity.

4. **f64 data type**: At 8 bytes per element, AVX2 provides 4 lanes per 256-bit vector register.

5. **Identity affine maps**: Input and output have identical layout. The default loop order (outermost to innermost dimension) already yields unit-stride access on the contiguous (last) dimension.

### Performance Bottleneck Hierarchy

For this class of element-wise, bandwidth-bound operations on the target Intel Xeon E5-2680 v4:

1. **SIMD vectorization** is the highest-impact single transformation. Without it, the core processes one f64 element per cycle; with AVX2, it processes 4. Since the operation is bandwidth-limited, wider loads/stores directly translate to higher throughput per core.

2. **Multi-core parallelism** is critical for large tensors. A single core saturates only a fraction of the available memory bandwidth. Distributing work across 28 cores (2 NUMA nodes × 14 cores) is essential to approach peak bandwidth, especially for tensors exceeding LLC capacity.

3. **Cache-aware tiling** matters for large tensors. When the working set far exceeds cache sizes, tiling the iteration space into cache-resident blocks improves spatial and temporal locality. For streaming workloads (read once, write once), the benefit is more modest than for compute-bound kernels, but tiling still helps by improving TLB hit rates, prefetcher effectiveness, and NUMA-local access patterns.

4. **Loop interchange** can further optimize access patterns after tiling, ensuring that the innermost traversal dimension has unit stride and maximizing spatial locality within tiles.

## Optimization Intents and Transformations

### Intent 1: SIMD Utilization (HIGH priority)

**Why**: The element-wise ReLU maps directly to SIMD compare-and-blend instructions. On AVX2 with f64, vectorization provides up to 4× throughput per core. Since the operation is bandwidth-bound, the wider load/store width is the primary benefit — fewer instructions to issue per data element, and the vector compare+select maps to efficient `vcmppd`/`vblendvpd` sequences.

**Transformations**:
- **Vectorization (Sequential Preprocessing)**: Tile the innermost loop(s) to the vector width using sequential tiling (`tile_using_for`), then lower the body to SIMD operations. This is the simplest vectorization path — it makes the innermost loop trip count match the vector width, enabling direct lowering.
- **Vectorization (Parallel Preprocessing)**: Tile the innermost loop(s) to the vector width using parallel tiling (`tile_using_forall`), which simultaneously distributes the outer tile loops across threads, then lower to SIMD. This combines vectorization with parallelization in a single action.

### Intent 2: Multi-Core Work Distribution (HIGH priority)

**Why**: All loop iterations are independent. On a 28-core machine with 2 NUMA nodes, leaving work on a single core wastes >96% of available compute and memory bandwidth. For large tensors (hundreds of MB to GB), parallelization is essential to approach peak memory bandwidth. For smaller tensors, the threshold for profitable parallelism depends on the overhead of thread creation/synchronization vs. the work per tile.

**Transformations**:
- **Parallelization (Tiling-based)**: Tile outer loop dimensions and distribute the resulting tiles across threads using `tile_using_forall`. The tile sizes control granularity — larger tiles reduce synchronization overhead but may cause load imbalance.
- **Parallelization (Thread-count-based)**: Directly partition the iteration space among a specified number of threads. The thread count must divide the iteration count to avoid remainder handling issues in downstream passes.

### Intent 3: Cache Locality (MEDIUM priority)

**Why**: The largest instances (e.g., 256×512×56×56 in f64 ≈ 3.2 GB) far exceed the per-core L1 (32 KB), L2 (256 KB), and even the shared LLC (~35 MB per socket). While element-wise ReLU is a streaming workload (each element read once and written once), tiling still benefits performance by: (a) improving hardware prefetcher effectiveness within tile boundaries, (b) reducing TLB pressure for large tensors, (c) enabling NUMA-aware data placement when combined with parallelization, and (d) creating tile-sized working sets that fit in L1/L2 for better pipeline utilization. For smaller instances that already fit in cache, tiling is less impactful but typically not harmful.

**Transformations**:
- **Tiling**: Tile the iteration space into blocks sized for L1 or L2 cache residency. For a streaming element-wise op, the relevant metric is that each tile's input + output footprint fits within the target cache level.
- **Loop Interchange**: Reorder loop dimensions to optimize memory access patterns within tiles. While the default order already provides unit-stride innermost access for row-major tensors with identity maps, interchange becomes relevant after tiling (where tile-loop vs. point-loop ordering affects locality) or if future fusion introduces non-trivial access patterns.
