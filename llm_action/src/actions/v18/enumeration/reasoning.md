# Layer 1 — Action Enumeration Reasoning (v18)

## Input Analysis

The input is a **matrix multiplication** kernel expressed as `linalg.matmul` in MLIR, operating on 2D tensors of `f64` type. The kernel template is:

```
%result = linalg.matmul ins(%A, %B : tensor<IxJxf64>, tensor<JxKxf64>)
                        outs(%C : tensor<IxKxf64>) -> tensor<IxKxf64>
```

A concrete instance uses `I=128, J=256, K=128`.

From a loop-nest perspective, `linalg.matmul` lowers to a triply-nested loop:
- Two parallel loops (over output rows and columns)
- One reduction loop (over the shared/contraction dimension)

The access patterns are:
- A is accessed as `A[i][j]` — row-major sequential in inner dim when j is innermost
- B is accessed as `B[j][k]` — row-major sequential in inner dim when k is innermost
- C is accessed as `C[i][k]` — row-major sequential in inner dim when k is innermost

## Target Hardware Summary

- **CPU**: Intel Xeon E5-2680 v4 (Broadwell)
- **Cores**: 28 physical (2 sockets x 14 cores), no hyperthreading
- **SIMD**: AVX2 + FMA (256-bit vectors, **no AVX-512**)
  - FP64: 4 elements per vector register
- **Cache**: L1d=32KB/core, L2=256KB/core, L3=~35MB shared per socket
- **NUMA**: 2 nodes

## Optimization Reasoning

### Intent 1: Data Locality and Reuse Optimization (HIGH)

**Why this is the top priority:**

Matrix multiplication has O(n^3) arithmetic on O(n^2) data, giving it high arithmetic intensity — but only if the data is resident in cache. Without tiling, the naive loop nest streams through entire matrices on each iteration of the outer loop, causing repeated cache evictions and memory traffic that far exceeds the minimum.

For the concrete 128x256x128 instance:
- A is 128x256 = 32K elements x 8 bytes = 256KB (exceeds L1d, fits in L2)
- B is 256x128 = 32K elements x 8 bytes = 256KB (exceeds L1d, fills L2)
- C is 128x128 = 16K elements x 8 bytes = 128KB

Total working set is ~640KB, which exceeds L2 (256KB per core). Tiling to fit tiles of all three operands into L1d or L2 is essential.

**Selected transformations:**

1. **Tiling**: The foundational transformation. Partitions the 3D iteration space into blocks so that sub-matrices of A, B, and C fit in cache. Multi-level tiling (e.g., L2-level outer tiles, L1-level inner tiles) is a standard technique in high-performance BLAS implementations.

2. **Loop Interchange**: The default loop order from `linalg.matmul` lowering may not match the optimal access pattern for the target memory layout. Interchanging loops to place the dimension with unit-stride access innermost (for vectorization-friendly access) and the reduction dimension at an appropriate level is critical. For example, an `i-k-j` ordering for row-major C and A with j innermost gives unit-stride access to both B and C.

3. **Promotion**: After tiling, sub-matrices may still have large strides in memory (e.g., accessing a 32x32 tile of a 256-wide matrix has stride 256 between rows). Copying tile data into compact, contiguous scratch buffers eliminates these strides, prevents cache set conflicts, and enables the vectorizer to assume aligned, sequential access. This is what high-performance BLAS libraries do with their internal packing routines.

### Intent 2: Compute Throughput Maximization (HIGH)

**Why this is equally critical:**

Even with perfect cache residency, the computation must saturate the CPU's functional units. Broadwell's AVX2+FMA can execute one 256-bit FMA per cycle, meaning 4 FP64 multiply-accumulate operations per cycle per core. Without vectorization, we use scalar FMA at best — a 4x throughput loss.

**Selected transformations:**

1. **Vectorization**: Maps the innermost loop to SIMD instructions. For FP64 matmul, the innermost loop (after interchange) should have unit-stride access and length divisible by 4 (the AVX2 FP64 lane count). This directly yields a 4x throughput improvement. Vectorization typically follows tiling and interchange, which prepare a suitable innermost loop.

2. **Loop Unrolling**: Unrolling an outer loop relative to the vectorized inner loop exposes multiple independent FMA streams. This is critical on Broadwell, where the FMA unit has 5-cycle latency but 1-cycle throughput — meaning at least 5 independent accumulation chains are needed to saturate the pipeline. Unrolling by 4-8 on the loop just outside the vectorized loop achieves this. Unrolling also amortizes loop overhead (branch, increment, compare).

### Intent 3: Coarse-Grain Parallelism (MEDIUM)

**Why MEDIUM rather than HIGH:**

For the concrete 128x128 output matrix, each thread would process very few tiles at 28 threads, and synchronization/scheduling overhead may dominate. However, the template is parameterized with variable `[I], [J], [K]`, and for larger problem sizes (512+), parallelization is essential for full machine utilization.

**Selected transformations:**

1. **Parallelization**: The outermost tile loop (after tiling) iterates over independent blocks of output rows/columns. These can be distributed across threads with no synchronization needed (each thread writes to a disjoint region of C). On 28 cores, this can provide up to 28x speedup for large problems, though memory bandwidth may become the bottleneck before reaching full linear scaling.

2. **Packing**: In a multi-threaded context, standard matrix layouts cause cache line sharing (false sharing) and TLB pressure when multiple threads access interleaved regions of the same matrix. Packing reorganizes data into thread-local, tile-contiguous panels. This is the technique used by GotoBLAS/OpenBLAS/BLIS: the B matrix is packed into column panels and the A matrix into row panels, each arranged for sequential access within the tile computation.

## Transformation Interaction Summary (informational, not part of output)

The typical optimization schedule for matrix multiplication on this hardware would be:
1. Tile (L2 level) → Interchange → Tile (L1/register level) → Promote → Vectorize → Unroll → Parallelize

However, this ordering information is explicitly **out of scope** for Layer 1. Layer 3 will determine valid composition sequences.

## Action Count Summary

- **3 intents** (2 HIGH, 1 MEDIUM)
- **7 transformations total** (3 + 2 + 2)
- All transformations are generic loop-nest operations applicable beyond matrix multiplication
- No kernel-specific dimension names used in action templates
