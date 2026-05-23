# Action Enumeration Reasoning — `linalg.add` (4D Element-wise Addition)

## Operation Analysis

The target operation is `linalg.add` on 4D tensors of shape `[A]x[B]x[C]x[D]xf64`. This is a purely element-wise operation: each output element is the sum of the two corresponding input elements. The operation manifests as a 4-deep perfectly nested loop with no reductions — every iteration is independent.

### Computational Characteristics

- **Arithmetic intensity**: Extremely low. One FP64 addition per 3 memory accesses (2 reads + 1 write). This is ~0.33 flops/double, far below the machine's compute-to-bandwidth ratio.
- **Data reuse**: None. Each input element is read exactly once; each output element is written exactly once. There is no temporal or spatial reuse across iterations in different dimensions.
- **Dependence structure**: All iterations are fully independent (parallel). No loop-carried dependencies of any kind.
- **Memory footprint**: For larger instances (e.g., 112x112x120x150 ≈ 226M elements), each tensor is ~1.8 GB in FP64. Three tensors means ~5.4 GB working set — far exceeding all cache levels.

### Performance Bottleneck

This operation is **purely memory-bandwidth-bound**. The CPU's arithmetic units are vastly underutilized; the bottleneck is the rate at which data can be streamed from/to DRAM through the memory hierarchy. Therefore, all optimizations should focus on:
1. Maximizing effective memory bandwidth utilization per core (vectorization, stride-1 access).
2. Scaling across all available cores (parallelization).
3. Organizing access patterns to be friendly to hardware prefetchers and TLBs (tiling, loop ordering).

### Shape Diversity

The dataset contains 271 instances with 4D shapes where each dimension ranges from small (7) to moderate (240). This means:
- The total element count varies by orders of magnitude.
- Some dimensions are very small (7, 14, 15) — parallelization and tiling must handle cases where a dimension has fewer elements than cores or cache line widths.
- The innermost dimension (D, stride-1) varies widely — vectorization may require peeling or masking for non-multiple-of-4 sizes.

## Intent and Transformation Rationale

### Intent 1: SIMD Vectorization for Bandwidth Saturation (HIGH)

Since `linalg.add` is bandwidth-bound, the single most impactful optimization is **vectorization**. Without SIMD, each core processes one FP64 element per cycle; with AVX2, it processes 4 elements per cycle via 256-bit vector loads, adds, and stores. This directly quadruples per-core throughput toward the bandwidth limit.

Effective vectorization requires the innermost loop to iterate over contiguous (stride-1) memory. For row-major 4D tensors, the last dimension (D) is stride-1. If the natural loop ordering doesn't place the stride-1 dimension innermost, **loop interchange** is needed as a prerequisite.

**Transformations selected:**
- **Vectorization**: Map the innermost contiguous loop to SIMD vector lanes. This is the single highest-impact transformation for this workload.
- **Loop Interchange**: Reorder the 4 nested loops so that the stride-1 dimension is innermost, enabling efficient vector memory operations and full cache line utilization. Even if the default ordering is already optimal, interchange remains a necessary action in the RL space since different tiling/parallelization decisions may alter which loop is innermost.

### Intent 2: Multi-Core Work Distribution (HIGH)

With 28 physical cores and fully independent iterations, parallelization is critical. A single core, even with perfect vectorization, can only consume a fraction of the total DRAM bandwidth (2 NUMA nodes). Distributing work across all cores is essential to approach peak memory bandwidth.

The prompt's few-shot examples specifically recommend implementing both tiling-based and num_threads-based parallelization as separate actions.

**Transformations selected:**
- **Parallelization (tiling-based)**: Partition the iteration space into tiles, then distribute tiles to threads via `scf.forall` or similar. This gives the RL agent control over tile granularity and naturally composes with tiling for cache optimization.
- **Parallelization (num_threads-based)**: Directly distribute outer loop iterations across a fixed number of threads. This is simpler but less flexible — good for cases where the outer dimension evenly divides the thread count.

### Intent 3: Cache and Streaming Efficiency (MEDIUM)

Although element-wise operations have no temporal data reuse, tiling still matters for large tensors:
- **TLB pressure**: With multi-GB working sets, the number of distinct virtual pages accessed simultaneously can exhaust TLB capacity, causing expensive page-table walks.
- **Hardware prefetch efficiency**: Modern CPUs track a limited number of prefetch streams. Tiling organizes access into a smaller number of concurrent streams, improving prefetch hit rates.
- **Cache line utilization**: Tiling ensures that when a cache line is fetched, all elements in it are consumed before eviction.

Loop unrolling helps with instruction-level parallelism (ILP): by unrolling the inner loop, the processor can overlap load latencies with computation, effectively software-pipelining the load-add-store sequence. This is particularly useful for bandwidth-bound code where memory latency (not bandwidth) can become the bottleneck at small working set sizes.

**Transformations selected:**
- **Tiling**: Block the iteration space to improve spatial locality, reduce TLB pressure, and organize prefetch-friendly access patterns. Tile sizes should target L1/L2 cache capacity.
- **Loop Unrolling**: Unroll inner loop iterations to increase ILP, reduce loop overhead, and enable the backend compiler to better schedule memory operations.

## Excluded Transformations and Rationale

- **Promotion / Packing**: Not useful. Element-wise addition has zero data reuse — copying data to contiguous local buffers adds overhead with no benefit.
- **Fusion**: Not applicable. The benchmark contains a single `linalg.add` operation with no producers or consumers to fuse with.
- **Im2col / Kernel-specific lowerings**: Not applicable. This is not a convolution or contraction.
- **Peeling**: While useful for handling vectorization remainder loops, peeling is typically handled implicitly by the vectorization action or the compiler backend rather than as a separate RL action.
