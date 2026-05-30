# Action Enumeration Reasoning — v40

## Benchmark Analysis

**Dataset**: `dataset_matmul` (train split, 187 instances)
**Operation**: `linalg.matmul` — C[I,K] += A[I,J] * B[J,K]
**Data type**: f64 (64-bit floating point)
**Shape range** (from sampled instances): dimensions span 128 to 1536, e.g., matmul_1024_1024_128, matmul_128_768_1024, matmul_1536_128_768, etc.

### Loop Nest Structure

`linalg.matmul` maps to a triply-nested loop:
- Two **parallel** iteration dimensions (I, K): output rows and columns.
- One **reduction** dimension (J): the contraction/inner-product axis.

This is the canonical matrix multiplication loop nest. Memory access patterns depend heavily on loop ordering: A[I,J] is row-major in I and stride-1 in J; B[J,K] is row-major in J and stride-1 in K; C[I,K] is the output accumulator.

### Target Hardware Considerations

- **Intel Xeon E5-2680 v4 (Broadwell)**: 28 physical cores, 2 NUMA nodes, no HT.
- **AVX2 + FMA**: 256-bit vectors → 4 FP64 lanes per vector register.
- **Cache hierarchy**: L1d ~32KB, L2 ~256KB, shared L3 (~35MB per socket).
- **No AVX-512**: must target 256-bit vector width.

### Working Set Analysis

For a matmul of dimensions I×J × J×K:
- Tile of A: I_tile × J_tile × 8 bytes
- Tile of B: J_tile × K_tile × 8 bytes
- Tile of C: I_tile × K_tile × 8 bytes

For even moderate shapes (e.g., 1024×1024×1024), the full operands are 8MB each — far exceeding any cache level. Tiling is essential to fit working sets into L1/L2.

### Key Performance Bottlenecks

1. **Memory bandwidth**: Without tiling, matmul is memory-bound due to repeated streaming of operands from DRAM. Cache blocking is the primary remedy.
2. **Spatial locality**: Loop order determines whether memory accesses are stride-1 (cache-line friendly) or strided (causing cache misses). Interchange can fix suboptimal orderings.
3. **SIMD underutilization**: Without vectorization, only scalar FMA units are used — leaving 75% of available FP64 throughput on the table (4 lanes vs 1).
4. **Single-core limitation**: Large matrices provide ample parallelism across outer loops. Not distributing across 28 cores wastes most of the machine.
5. **Conflict misses and TLB pressure**: Even with tiling, non-contiguous access to tiled slices of large matrices can cause conflict misses. Promotion (copying tiles into contiguous scratch buffers) addresses this.

## Intent & Action Design Rationale

### Intent 1: Data Locality & Cache Efficiency (HIGH)

This is the highest-impact optimization category for matmul. The triply-nested loop with large iteration spaces creates working sets that exceed cache by orders of magnitude. All three transformations in this intent address different facets of the data locality problem.

- **Tiling**: The foundational transformation. Partitions the I×J×K iteration space into blocks sized to fit in L1/L2 cache. Directly reduces cache miss rate and converts a memory-bound computation into a compute-bound one. The `parallelize` mode additionally enables tiling-based work distribution across threads.
- **Loop Interchange**: Ensures the innermost loop iterates over the dimension with stride-1 memory access. For row-major matmul, this means the innermost loop should sweep over K (columns of C and B) rather than J or I. Incorrect loop ordering can degrade performance by an order of magnitude.
- **Promotion**: After tiling, operand tiles are subviews of the original large matrices and may not be contiguous in memory. Copying these tiles into compact, aligned temporary buffers eliminates conflict misses, reduces TLB pressure, and provides the contiguous data layout that vectorized inner kernels require. Operates at buffer (memref) level and includes internal bufferization.

### Intent 2: Compute Throughput Maximization (HIGH)

Once data fits in cache, performance depends on how efficiently the CPU's execution units are utilized. This intent targets SIMD exploitation, instruction-level parallelism, and multi-core scaling.

- **Vectorization**: Maps the innermost loop dimension to AVX2 vector lanes (4 FP64 elements). This is critical for approaching peak FLOPS — without it, only 1/4 of available FMA throughput is used. Vectorization is preceded by a tiling preprocessing step that shapes the target loop band to the vector width.
- **Unrolling**: Unrolls loop iterations to expose multiple independent FMA operations to the out-of-order execution engine. Reduces loop branch overhead, increases register utilization, and helps the scheduler hide memory latency. Particularly effective on the register-blocked inner kernel after tiling and vectorization.
- **Parallelization**: Distributes outer-loop iterations across the 28 physical cores. Matmul's two parallel dimensions (I and K) provide embarrassingly parallel partitioning opportunities. This is the `num_threads`-based standalone parallelization (distinct from tiling's built-in `parallelize` mode).
