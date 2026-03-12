# Action Enumeration Reasoning — v4

## Analysis Context

The input consists of three kernel types expressed as MLIR `linalg` structured operations wrapped in timing harnesses:

1. **Matrix Multiplication** (`linalg.matmul`): A rank-2 contraction with iteration space [I, J, K] where I and K are parallel and J is a reduction dimension. Concrete instance: 256x512 @ 512x1024.

2. **2D Convolution** (`linalg.conv_2d_nchw_fchw`): A sliding-window contraction with 7 loop dimensions (batch N, output channels F, output spatial OH/OW, input channels C, kernel spatial KH/KW). Concrete instance: batch=128, C=32, H=W=7, F=256, KH=KW=1.

3. **Element-wise Generic** (`linalg.generic`): A 5D parallel element-wise operation (addition). Concrete instance: 8x8x16x8x32. All iterator types are parallel — no reduction dimension.

All operations use `memref` (pre-bufferized) types with `f64` element type, targeting Intel Xeon E5-2680 v4 (Broadwell, AVX2, 28 cores, 2 NUMA nodes).

## Key Observations Driving the Enumeration

### Compute vs. Memory Characteristics
- **Matmul** is compute-bound with O(I*J*K) FLOPs and O(I*J + J*K + I*K) data. High arithmetic intensity means tiling for register/cache reuse and vectorization for FMA throughput are paramount.
- **Conv2D** is also compute-bound but with a more complex iteration space. The sliding-window access pattern introduces reuse opportunities across output spatial dimensions. With KH=KW=1, this particular instance degenerates toward a batched matmul.
- **Generic (element-wise)** is memory-bound with O(N) FLOPs and O(N) data. Performance is dominated by memory bandwidth, making vectorization (to maximize load/store throughput) and parallelization (to utilize aggregate bandwidth across NUMA nodes) the primary levers.

### Hardware Constraints (Broadwell AVX2)
- FP64 vector width: 4 elements (256-bit AVX2).
- FMA available: 2 FLOPs/element/cycle → peak throughput requires keeping FMA pipeline fed.
- Cache hierarchy: 32KB L1d, 256KB L2, ~35MB shared L3 per socket.
- 28 cores, 2 NUMA nodes → coarse parallelism over outer loops is beneficial for large problems.

### Loop-Nest Perspective
All three kernels are regular loop nests over rectangular iteration spaces with affine indexing. This makes them ideal targets for classical loop transformations: tiling, interchange, vectorization, unrolling, parallelization, and data layout optimization.

## Intent Design Rationale

### Intent 1: Data Locality and Cache Utilization (HIGH)
For compute-bound kernels (matmul, conv), the gap between cache-hit and cache-miss execution can be 10-100x. Tiling is the foundational transformation. Multi-level tiling maps to the L1/L2/L3 hierarchy. Promotion eliminates conflict misses after tiling. Packing transforms data layout to match tiled access patterns. Even for memory-bound kernels (generic), tiling can improve streaming behavior and TLB utilization.

### Intent 2: SIMD Exploitation (HIGH)
AVX2 FP64 provides 4x throughput over scalar. Without vectorization, we leave 75% of peak performance on the table. Vectorization requires appropriate innermost loop ordering (interchange), sufficient independent iterations (unrolling to fill FMA latency), and aligned trip counts (peeling). These are tightly coupled but remain separate macro actions since they are independently applicable.

### Intent 3: Thread-Level Parallelism (MEDIUM)
With 28 cores, parallelism over outer loops can provide up to 28x speedup. However, it requires careful grain sizing to avoid oversubscription and false sharing. For element-wise operations, parallelism is critical since bandwidth scales with core count. For matmul/conv, parallelism complements tiling. Fusion is included here because it affects what can be parallelized together and reduces inter-loop synchronization.

### Intent 4: Iteration Space Restructuring (MEDIUM)
Some transformations reshape the problem structure to enable other optimizations. Decomposition (e.g., im2col for convolutions) converts complex access patterns into regular ones. Padding aligns dimensions for vectorization and tiling. Generalization exposes all loop dimensions for unrestricted transformation. Canonicalization maintains IR hygiene between transformation steps.

### Intent 5: Register-Level Optimization (MEDIUM)
After tiling and vectorization, the innermost computation operates on small blocks that should reside in registers. Loop unrolling exposes independent FMA chains to hide latency. Unroll-and-jam (also known as register tiling) is a distinct strategy that unrolls an outer loop and fuses the copies, creating multiple independent accumulation streams. Software pipelining reorders instructions to overlap loads with computation. These are fine-grained but critical for achieving peak throughput on Broadwell.

## Transformation Selection Principles

1. **No kernel-specific specialization**: All transformations are described in loop-nest terms, applicable to any of the three kernel types.
2. **RL-action granularity**: Each transformation is a single discrete action with parameters determined by Layer 2.
3. **No compound actions**: Tiling, interchange, vectorization, etc. are separate actions even though they often compose.
4. **No parameter values or legality**: Ranges, constraints, and preconditions are deferred to Layer 2.
5. **Coverage**: The enumeration covers data movement (tiling, promotion, packing), compute throughput (vectorization, unrolling), parallelism (parallelization, fusion, distribution), and structural transformations (decomposition, padding, generalization, canonicalization).
