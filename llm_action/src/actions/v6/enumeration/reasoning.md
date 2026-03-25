# Layer 1 — Action Enumeration Reasoning (v6)

## Input Analysis

The input is a `linalg.matmul` operation on tensors with shape parameters `[I]x[J]` @ `[J]x[K]` -> `[I]x[K]` using `f64` elements. Concrete instances include shapes like `128x256 @ 256x128 -> 128x128`. This is a classic three-nested-loop computation: two parallel loops (over I and K) and one reduction loop (over J).

## Target Hardware Considerations

- **Intel Xeon E5-2680 v4 (Broadwell)**: 28 cores, 2 NUMA nodes, AVX2+FMA (no AVX-512).
- **FP64 vector width**: 4 lanes (256-bit AVX2).
- **Cache hierarchy**: L1d ~32KB, L2 ~256KB, L3 shared per socket (~tens of MB).
- Hyper-threading disabled; 1 thread per core.

## Optimization Reasoning

For a dense matrix multiplication loop nest on this hardware, the dominant performance factors are:

### 1. Data Locality via Tiling (HIGH priority)
Matrix multiplication is memory-bandwidth-sensitive at scale. The three-loop nest accesses three matrices with different stride patterns. Without tiling, the working set overflows caches quickly, causing excessive DRAM traffic. Multi-level tiling (L1/L2-aware blocking) is the single most impactful transformation for matmul on CPUs. This is universally the first optimization applied in high-performance BLAS implementations.

### 2. SIMD Exploitation via Vectorization (HIGH priority)
AVX2+FMA provides 4-wide FP64 vector operations. The innermost loop must be mapped to SIMD lanes to exploit this hardware. Without vectorization, the kernel runs at 1/4 of peak throughput. Vectorization interacts with loop ordering — the innermost loop should access contiguous memory for efficient vector loads/stores. This is critical for achieving near-peak FLOPS.

### 3. Loop Ordering via Interchange (MEDIUM priority)
The default loop ordering of `linalg.matmul` may not be optimal for cache line utilization. Permuting loops can change memory access patterns from strided to sequential, improving spatial locality and enabling more efficient vectorization. For matmul specifically, the classical optimization is ensuring the innermost loop accesses contiguous memory in the output and one input matrix. However, this is shape-dependent — for some shapes the default ordering is already adequate.

## Action Selection Rationale

I selected three transformations that form the core optimization repertoire for dense loop nests on CPUs:

1. **Tiling** — the foundational cache-blocking transformation; essential for any non-trivial matrix size.
2. **Vectorization** — mandatory to exploit SIMD hardware; without it, peak throughput is unreachable.
3. **Loop Interchange** — enables better access patterns and can be critical for enabling efficient vectorization and cache utilization. Placed at MEDIUM because its benefit depends on the specific shape and existing loop ordering.

These three are independent macro actions that compose well and cover the primary optimization axes for dense loop nests on CPU: data reuse (tiling), instruction-level parallelism (vectorization), and memory access pattern optimization (interchange).

I deliberately excluded parallelization, packing, and unrolling at this stage:
- **Parallelization**: Important for multi-core but secondary to getting single-core performance right first.
- **Packing/Layout**: Typically applied after tiling to improve micro-kernel data layout; secondary.
- **Unrolling**: A micro-optimization that follows after the major structural transforms are in place.

These could be added in a future enumeration iteration as the action space matures.
