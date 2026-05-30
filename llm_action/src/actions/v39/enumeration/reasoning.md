# Action Enumeration Reasoning — v39

## Benchmark Analysis

The dataset_ml benchmark contains **1135 instances** across 5 operation families, targeting **Intel Xeon E5-2680 v4** (Broadwell, 28 physical cores, 2 NUMA nodes, AVX2+FMA, no AVX-512, f64 → 4 SIMD lanes).

### Operation Family Profiles

| Family | Count | Loop Depth | Parallel Dims | Reduction Dims | Compute Profile |
|--------|-------|-----------|---------------|----------------|-----------------|
| conv_2d_nchw_fchw | 277 | 7 (N,F,OH,OW,C,KH,KW) | N,F,OH,OW | C,KH,KW | Compute-bound |
| pooling_nchw_max | 249 | 6 (N,C,OH,OW,KH,KW) | N,C,OH,OW | KH,KW | Mixed (small kernels → bandwidth-sensitive) |
| add | 270 | 4 (all parallel) | All 4 dims | None | Memory-bandwidth bound |
| matmul | 186 | 3 (I,K,J) | I,K | J | Compute-bound |
| relu | 148 | 2-4 (all parallel) | All dims | None | Memory-bandwidth bound |

### Hardware Constraints Driving Optimization Choices

- **AVX2+FMA (no AVX-512)**: 4 × f64 per vector instruction → vectorization provides up to 4× throughput per core
- **28 physical cores (2×14)**: work distribution across cores is essential; avoid oversubscription
- **Cache hierarchy**: L1d 32KB, L2 256KB, L3 ~35MB per socket → tiling to cache sizes is critical for compute-bound kernels
- **NUMA topology**: 2 sockets with separate memory controllers → coarse partitioning benefits large tensors

### Key Optimization Observations

1. **Compute-bound kernels (matmul, conv)**: These have high arithmetic intensity but only realize it if working sets fit in cache. Tiling is the dominant optimization. Matmul has clean access patterns that tile naturally. Convolution has more complex strided access patterns that benefit from either careful tiling+interchange or structural lowering (im2col) to matmul-like form.

2. **Bandwidth-bound kernels (add, relu)**: These process each element once with minimal arithmetic. The bottleneck is memory bandwidth, not compute. Parallelization to spread traffic across cores and vectorization to increase per-cycle data consumption are the primary levers.

3. **Mixed kernels (pooling)**: Small kernel sizes (1×1 to 3×3) mean limited reduction work per output element. Closer to bandwidth-bound, but with enough arithmetic for vectorization to help.

4. **All kernels have parallel loops**: Every operation family has at least 2 parallel loop dimensions, making parallelization universally applicable.

## Intent Structure

Three intents, all HIGH priority, covering the three orthogonal axes of CPU performance:

### Intent 1: Data Locality and Cache Exploitation (HIGH)
**Why HIGH**: For matmul and conv (463 instances, 41% of benchmark), cache-aware tiling determines whether peak arithmetic throughput is achievable. Matmul's O(n³) compute on O(n²) data has theoretical reuse that only materializes with proper blocking. Conv's 7-deep loop nest with reduction over C,KH,KW needs tiling to keep filter and input windows in cache.

**Transformations**: Tiling (fundamental blocking), Loop Interchange (stride optimization), Promotion (contiguous buffer copies for strided slices).

### Intent 2: Compute Throughput Maximization (HIGH)
**Why HIGH**: AVX2+FMA provides 4× throughput multiplier for f64. Without vectorization, the processor uses only 1/4 of its ALU capacity. This applies to all 1135 instances — compute-bound kernels need it for FLOPS, bandwidth-bound kernels need it to consume data faster per cycle.

**Transformations**: Vectorization with sequential tiling (standard SIMD path), Parallel Vectorization with parallel tiling (combined distribution+SIMD path for bandwidth-bound ops), Loop Unrolling (ILP exposure and loop overhead reduction).

### Intent 3: Multi-Core Parallelism and Computation Restructuring (HIGH)
**Why HIGH**: 28 cores provide ~28× potential throughput. Every operation family has distributable parallel dimensions. Additionally, im2col restructuring benefits the 277 convolution instances by converting irregular access patterns to regular matmul-like form.

**Transformations**: Tiling-Based Parallelization (flexible work granularity), Direct Parallelization (simpler, for evenly divisible dimensions), Im2col Lowering (structural transformation for convolution family).

## Transformation Count: 9

Distributed as 3 per intent. Each transformation is a unique macro RL action with parameters deferred to Layer 2. The only kernel-specific transformation is Im2col Lowering, which is explicitly permitted by the system description and affects 24% of the benchmark set.
