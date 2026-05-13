# Action Enumeration Reasoning — v26

## Benchmark Set Analysis

The training set contains 20 instances across 5 operation families, all using f64 data type:

### Operation Families

1. **matmul** (4 instances): Classic 2D matrix multiplication with 3 loop dimensions (2 parallel: M, N; 1 reduction: K). Sizes range from 128x256x128 to 256x2048x512. Compute-bound; arithmetic intensity scales with dimension sizes. The quintessential dense linear algebra kernel.

2. **conv_2d_nchw_fchw** (4 instances): 2D convolution with NCHW/FCHW layout, producing a 7-deep loop nest (4 parallel: N, F, OH, OW; 3 reduction: C, KH, KW). Strides vary (1 or 2). Kernel sizes range from 1x1 (pointwise) to 3x3. Compute-bound with complex, multi-dimensional data access patterns and potential for stride irregularities.

3. **pooling_nchw_max** (4 instances): Max pooling with 6-deep loop nest (4 parallel: N, C, OH, OW; 2 reduction: KH, KW). Spatial sizes range from 14x14 to 224x224 with various kernel and stride configurations. Less compute-dense than conv2d but still has reduction loops and sliding-window access.

4. **add** (4 instances): Elementwise 4D tensor addition with 4 fully parallel loop dimensions. Purely memory-bound — performance is limited by memory bandwidth, not compute. Shapes vary from small (112x15x15x15) to large (120x120x228x130).

5. **relu** (4 instances): Elementwise ReLU via linalg.generic with all-parallel iterator types. Variable rank (2D and 4D instances). Also memory-bound. Involves a comparison and select per element.

### Hardware Context

Target: Intel Xeon E5-2680 v4 (Broadwell)
- 28 physical cores (2 sockets x 14), no SMT
- AVX2 + FMA (no AVX-512) → 4 f64 lanes per 256-bit vector register
- Cache: 32KB L1d, 256KB L2 per core, shared L3 per socket
- 2 NUMA nodes

### Key Performance Bottlenecks

**Compute-bound operations (matmul, conv2d):**
- Working sets far exceed per-core caches (e.g., matmul 256x2048x512 in f64 = ~4GB total)
- Peak throughput requires tiling to L1/L2, vectorization to AVX2, and loop ordering for unit-stride access
- FMA units need unrolled, vectorized inner loops to stay saturated

**Memory-bound operations (add, relu):**
- Performance limited by memory bandwidth (~60-80 GB/s aggregate)
- Vectorization is essential to maximize bytes/cycle throughput
- Parallelization across cores scales bandwidth utilization
- Tiling still helps for cache-line efficiency and NUMA awareness

**Sliding-window operations (conv2d, pooling):**
- Multi-dimensional access patterns with potential stride irregularities
- Conv2d benefits from im2col lowering to convert to matmul-like form
- Loop interchange is critical to find optimal traversal order

## Intent Organization Rationale

The 4 intents are organized along orthogonal optimization axes:

1. **Data Locality & Cache Optimization** (HIGH): Groups transformations that address data movement — tiling creates cache-sized blocks, promotion copies them to contiguous buffers, packing rearranges the data layout. These are foundational for compute-bound operations and beneficial for all.

2. **SIMD Vectorization & Instruction Efficiency** (HIGH): Groups transformations that maximize per-core throughput — vectorization maps to AVX2, interchange ensures unit-stride innermost loops, unrolling saturates execution units. Critical for all operations since f64 vectorization provides up to 4x throughput.

3. **Coarse-Grain Parallelism & Work Distribution** (MEDIUM): Groups transformations that utilize the 28-core topology — two parallelization strategies (tiling-based and thread-count) plus split reduction for enabling parallel accumulation. Rated MEDIUM because single-core optimization (Intents 1 & 2) typically has higher per-operation impact, but parallelization is necessary for utilizing the full machine.

4. **Iteration Space Restructuring & IR Normalization** (MEDIUM): Groups enabling/cleanup transformations — im2col converts conv2d to matmul-like form, peeling handles remainders for clean vectorization, canonicalization normalizes IR between transformation steps. These are prerequisites or facilitators for the other intents.

## Transformation Coverage

All 12 transformations cover the fundamental optimization space for dense loop nests on this hardware:
- 3 data locality transforms (Tiling, Promotion, Packing)
- 3 vectorization/ILP transforms (Vectorization, Loop Interchange, Loop Unrolling)
- 3 parallelism transforms (Tiling-based Parallelization, Thread-count Parallelization, Split Reduction)
- 3 restructuring transforms (Im2col Lowering, Loop Peeling, Canonicalization)

No transformation is duplicated across intents, and each represents a distinct, independently parameterizable RL macro action.
