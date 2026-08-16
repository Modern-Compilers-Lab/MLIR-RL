# Action Enumeration Reasoning — v55

## Benchmark Analysis

The benchmark set `dataset_ml` (train split, 1135 instances) contains five operation families representative of machine-learning inference/training workloads on CPUs:

### Operation Families

1. **conv_2d_nchw_fchw** (277 instances): 2D convolution in NCHW input / FCHW filter layout. This produces a 7-deep nested loop structure (N, F, C, OH, OW, KH, KW) with sliding-window access patterns. Compute-bound with O(N*F*C*OH*OW*KH*KW) arithmetic operations. The sampled shapes show batch sizes {128, 256}, channel counts ranging from 32 to 512, spatial sizes from 7 to 112, kernel sizes {1x1, 3x3, 7x7}, and strides of 1 or 2. The diversity of shapes (1x1 pointwise vs. 3x3 spatial vs. large spatial) means tile sizes and strategy must be parameterized.

2. **matmul** (186 instances): Matrix multiplication with a 3-deep nested loop structure (I, J, K — two parallel, one reduction). Compute-bound with O(I*J*K) operations. Dimensions range from 128 to 3072 in various combinations. This is the canonical target for cache-blocking, vectorization, and packing optimizations. The wide range of aspect ratios (e.g., 3072x512x128 vs. 128x128x1024) means transformations must adapt to shape.

3. **pooling_nchw_max** (249 instances): Max pooling in NCHW layout with a 6-deep loop nest (N, C, OH, OW, KH, KW). Contains a reduction (max) over the pooling window. Less compute-intensive than convolution but still benefits from tiling and vectorization. Shapes show pooling windows from 1x1 to 7x7 and spatial sizes from 7 to 240. Strides vary (1, 2, or derived from output shape).

4. **add** (270 instances): 4D elementwise addition. All four loops are embarrassingly parallel. Memory-bound: performance is limited by memory bandwidth, not compute. Tensor dimensions vary widely (7 to 240 per dimension). The key optimization levers are vectorization (to maximize bandwidth utilization per core) and parallelization (to aggregate bandwidth across cores/sockets).

5. **relu** (148 instances): Elementwise ReLU via `linalg.generic` with a compare-and-select pattern. Both 2D (128x1024) and 4D (128x384x28x28) shapes appear. Like add, this is memory-bound with all-parallel loops. Same optimization strategy applies: vectorize and parallelize for bandwidth.

### Hardware Context

Target: **Intel Xeon E5-2680 v4 (Broadwell)**
- 28 physical cores (2 sockets x 14 cores), no SMT
- AVX2 + FMA: 256-bit vectors → **4 FP64 lanes** per vector
- No AVX-512 (do not assume 512-bit operations)
- L1d: 32KB/core, L2: 256KB/core, L3: shared per socket (~35MB)
- 2 NUMA nodes

### Key Observations for Action Space Design

1. **Compute vs. memory bound dichotomy**: matmul and conv2d are compute-bound; add, relu, and (to a lesser extent) pooling are memory-bound. The action space must cover both compute-optimization strategies (tiling for reuse, vectorization for throughput, packing for stride elimination) and bandwidth-optimization strategies (parallelism for bandwidth aggregation, vectorization for per-core bandwidth).

2. **Shape diversity requires parameterization**: Dimensions range from 7 to 3072. Fixed tile sizes or parallelization factors would be suboptimal for most shapes. Every transformation must be parameterized so the RL agent can learn shape-dependent strategies.

3. **Convolution structural complexity**: Conv2d's 7 nested loops with sliding-window access create complex tiling and vectorization decisions. Im2col lowering can convert convolution to a matmul-like contraction, enabling reuse of optimized matmul schedules.

4. **All operations have parallel dimensions**: Batch (N), output channels (F/C), spatial outputs (OH, OW), and all elementwise dimensions are parallel — the 28 cores can always be utilized.

5. **FP64 with AVX2**: 4 lanes per vector. Without vectorization, only 25% of peak scalar throughput is used. Vectorization is universally beneficial.

6. **Cache hierarchy sizing**: L1d (32KB) fits ~4K f64 values; L2 (256KB) fits ~32K f64 values. For matmul with I=J=K=1024, the full operands are 8MB each — far exceeding per-core caches. Tiling to fit working sets in L1/L2 is essential.

## Intent and Transformation Selection Rationale

### Intent 1: Cache Locality and Data Reuse (HIGH)

The most impactful optimization family for compute-bound kernels. Without tiling, matmul and conv2d suffer from poor data reuse — each element is loaded from main memory for every use rather than being reused from cache.

- **Tiling**: The foundational transformation. Partitions the iteration space so working sets fit in L1/L2. For matmul, tiling the I, J, K loops creates tiles where operand sub-matrices are reused O(tile_size) times from cache.
- **Loop Interchange**: Determines which dimension is innermost, directly controlling whether memory accesses are stride-1 (contiguous) or strided. Critical for getting vectorization-friendly access patterns.
- **Promotion**: After tiling, sub-tensors may have non-unit strides in memory (e.g., a column tile of a row-major matrix). Promoting operands to contiguous local buffers eliminates stride waste, reduces conflict misses, and enables clean vectorization.

### Intent 2: SIMD Exploitation (HIGH)

AVX2 provides 4x FP64 throughput via vector FMA instructions. All five operation families benefit — compute-bound ops get arithmetic throughput, memory-bound ops get bandwidth utilization.

- **Vectorization (Sequential Preprocessing)**: The standard path: tile innermost loops to vector width using `tile_using_for`, then lower to SIMD. Outer tiles remain sequential. This is appropriate when parallelism is handled by a separate action.
- **Vectorization (Parallel Preprocessing)**: An alternative path using `tile_using_forall` for preprocessing tiling. This simultaneously sets up vector-width inner tiles AND distributes outer tiles across threads. Useful when the RL agent wants to combine vectorization and parallelization in one step.
- **Unrolling**: After vectorization, unrolling the next-innermost loop exposes multiple independent vector operations to the out-of-order engine, hiding FMA latency (~5 cycles on Broadwell). For memory-bound ops, unrolling helps issue multiple vector loads per iteration to saturate bandwidth.

### Intent 3: Coarse-Grain Parallelism (HIGH)

Without parallelization, only 1 of 28 cores is used — wasting >96% of the machine. All operation families have parallel outer loops.

- **Parallelization (Tiling-based)**: Creates parallel work units via tiling with `forall` semantics. Gives fine control over granularity: tile sizes determine work per thread. Preferred when iteration counts are irregular.
- **Parallelization (Thread-based)**: Directly distributes iterations across N threads. Simpler but requires iteration count divisible by thread count. Works well for regular, large iteration spaces (e.g., batch dimension = 128 or 256).

### Intent 4: Iteration Space and Kernel Restructuring (MEDIUM)

Structural transformations that change the form of the computation to expose better optimization opportunities downstream.

- **Im2col Lowering**: Convolution-specific but widely applicable (277 instances). Converts the 7-loop convolution into a matmul-like contraction, enabling reuse of optimized matmul schedules. The copy overhead is typically amortized by the compute savings.
- **Packing**: Restructures data layout (tiling + transposing dimensions) so inner tiles are contiguous in memory. Distinct from promotion: packing changes the logical-to-physical data mapping, while promotion copies to local buffers. Essential for eliminating non-unit-stride vector accesses.

## Summary

The enumerated action space contains 4 intents with 10 total transformations:
- 3 HIGH-priority intents covering the fundamental optimization axes (cache, SIMD, parallelism)
- 1 MEDIUM-priority intent for structural transformations (im2col, packing)

This provides the RL agent with a complete set of macro actions that can be composed into effective optimization schedules for all five operation families in the benchmark set. The parameterization of each action (tile sizes, vector widths, thread counts, operand indices, etc.) gives Layer 2 clear guidance for implementing tunable, RL-friendly actions.
