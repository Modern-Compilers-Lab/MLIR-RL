# Action Enumeration Reasoning — v41 (dataset_matmul)

## Input Analysis

The benchmark set consists of 187 matmul instances of the form `linalg.matmul` operating on FP64 tensors:
- Input A: `tensor<I×J×f64>`, Input B: `tensor<J×K×f64>`, Output C: `tensor<I×K×f64>`
- The operation is a triply-nested loop with two parallel dimensions (I, K) and one reduction dimension (J).
- Shapes vary across a wide range (128–2048 per dimension), producing working sets from ~0.4 MB to ~96 MB — well exceeding L1 (32 KB), L2 (256 KB), and often L3 capacity.

## Target Hardware Constraints

Intel Xeon E5-2680 v4 (Broadwell):
- 28 physical cores, 2 NUMA nodes, no SMT
- AVX2 + FMA (no AVX-512): 4 FP64 lanes per 256-bit register
- Cache: L1d 32 KB, L2 256 KB, shared L3 ~35 MB
- Peak FP64 throughput per core: 2 FMA units × 4 lanes = 8 FLOPS/cycle

## Optimization Strategy

For dense matmul on this hardware, three categories of optimization are essential:

### 1. Data Locality & Cache Efficiency (HIGH priority)

Matmul's O(N³) compute on O(N²) data means significant data reuse is possible, but only if the iteration space is partitioned so working sets fit in cache. Three transforms address this:

- **Tiling**: The foundational transform. Partitions I×J×K iteration space into blocks where the A-tile (I_t × J_t), B-tile (J_t × K_t), and C-tile (I_t × K_t) fit simultaneously in L1 or L2. Without tiling, the entire B matrix is streamed for each row of A — catastrophic for large sizes.

- **Loop Interchange**: The reduction dimension J can appear in any loop position. Placing K (the stride-1 dimension for both B and C in row-major layout) innermost and J outermost maximizes spatial locality. Wrong ordering can cause cache misses on every element access.

- **Promotion**: After tiling, operand subviews are non-contiguous slices of large matrices. Copying them into compact aligned temporary buffers eliminates set-associativity conflict misses, reduces TLB entries, and provides the dense contiguous layout that vectorized inner kernels require for peak throughput. Requires bufferization as a preprocessing step.

### 2. Compute Throughput Maximization (HIGH priority)

Once data fits in cache, performance depends on how efficiently FP64 execution units are utilized:

- **Vectorization (Sequential Tiling)**: Maps inner loop iterations to AVX2 FMA vector lanes (4× FP64 throughput). Sequential tiling preprocessing (tile_using_for) shapes loop bounds to vector widths without introducing parallelism — a pure SIMD transform.

- **Vectorization (Parallel Tiling)**: Same SIMD lowering, but the preprocessing tiling uses tile_using_forall, which distributes outer tiles across threads while vectorizing inner tiles. This fuses parallelization with vectorization in one action.

- **Unrolling**: After tiling and vectorization, the inner loops may still underutilize the out-of-order engine. Unrolling exposes multiple independent FMA instructions, enabling the CPU to overlap execution across FMA units and hide arithmetic latency. Also reduces loop-control overhead for tight inner kernels.

### 3. Multi-Core Parallelism & Work Distribution (HIGH priority)

With 28 cores, single-threaded execution wastes 96% of available compute:

- **Parallelization (Tiling-Based)**: Uses tile_using_forall to partition the iteration space into coarse tiles distributed across threads. Tile sizes control work granularity and can be tuned for load balance.

- **Parallelization (Thread-Count-Based)**: Directly distributes outer loop iterations across a fixed thread count. Simpler than tiling-based parallelization when loop bounds provide natural parallelism. Requires iteration count divisible by thread count.

- **Packing**: Reorganizes operand data from standard row/column-major into a blocked tile-contiguous layout. In parallel execution, this eliminates false sharing between threads, ensures each thread's working set is spatially compact, and reduces TLB pressure. Classic technique from BLIS/GotoBLAS for high-performance parallel matmul.

## Design Decisions

1. **Vectorization split into two actions** (sequential vs parallel preprocessing): Per the system guidelines, these represent fundamentally different code generation paths — one purely SIMD, the other fusing SIMD with thread distribution.

2. **Parallelization split into two actions** (tiling-based vs num_threads-based): These offer different trade-offs in granularity control and simplicity, both worth exploring in the RL action space.

3. **Packing as a separate action from Promotion**: Promotion copies data into temporary buffers maintaining the original sub-layout; Packing restructures the data layout itself into tile-contiguous form. Both address memory efficiency but through different mechanisms.

4. **All three intents at HIGH priority**: For matmul on a 28-core Broadwell with AVX2, all three categories (locality, SIMD, parallelism) are non-negotiable for competitive performance. Omitting any one category leaves massive performance on the table.
