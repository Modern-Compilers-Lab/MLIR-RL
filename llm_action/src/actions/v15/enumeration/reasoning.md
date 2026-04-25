# Layer 1 — Action Enumeration Reasoning (v15)

## Input Analysis

The RL system input is a single `linalg.conv_2d_nchw_fchw` operation — a 2D convolution with NCHW input layout and FCHW filter layout.

### Loop Nest Structure

From a loop-nest perspective, this operation expands to a 7-deep nested loop:
- **4 parallel loops**: iterating over batch, output-channel, output-height, output-width dimensions of the output tensor.
- **3 reduction loops**: iterating over input-channel, kernel-height, kernel-width dimensions, accumulating the convolution result.

The body of the innermost loop performs a multiply-accumulate (FMA) operation.

### Data Access Patterns

- **Input tensor** (`tensor<N x C x H x W>`): accessed with sliding-window strides along spatial dimensions and full traversal along the channel reduction dimension. Reuse exists across output spatial positions (overlapping windows) and across output channels.
- **Filter tensor** (`tensor<F x C x KH x KW>`): accessed fully for each output spatial position. Reuse exists across batch and output spatial positions.
- **Output tensor** (`tensor<N x F x OH x OW>`): each element is written once, accumulated across all reduction loops.

### Key Performance Characteristics

1. **Compute intensity**: High for large channel/filter counts, potentially memory-bound for small kernel sizes (e.g., 1x1 convolutions).
2. **Data reuse**: Significant reuse across multiple dimensions — exploiting this via tiling/packing is critical.
3. **Irregular strides**: NCHW layout means spatial dimensions are innermost in memory, but channel dimensions introduce large strides when accessed in reduction loops. Packing can regularize these.

## Target Hardware Considerations

**Intel Xeon E5-2680 v4 (Broadwell)**:
- 28 physical cores, 2 NUMA nodes, no HT
- AVX2 + FMA (256-bit vectors: 4 FP64 lanes)
- L1d ~32KB, L2 ~256KB per core, shared L3 per socket
- No AVX-512

### Implications for Optimization

- **Tiling is essential**: Working sets for the 7-deep loop nest easily exceed cache sizes. Multi-level tiling (L1/L2-aware) is the primary performance lever.
- **Vectorization must target AVX2**: 4 FP64 lanes per vector. One parallel dimension should be mapped to the vector width.
- **Parallelism across 28 cores**: Outer parallel loops (batch, output-channel, output spatial) provide natural parallelism. Must avoid oversubscription.
- **Register pressure**: 16 YMM registers. Aggressive unrolling risks spills — moderate unroll factors preferred.
- **Memory layout matters**: NCHW layout creates stride-1 access only along the innermost (width) dimension. Packing into contiguous tiles significantly helps.

## Optimization Strategy

### Intent 1: Cache Locality and Data Reuse (HIGH priority)

This is the most impactful optimization category for a 7-deep convolution loop nest with three tensor operands.

**Rationale**: The total data footprint of the convolution far exceeds per-core cache sizes. Without tiling, every inner-loop iteration incurs cache misses for at least one operand. Tiling partitions the iteration space so that working sets fit in L1 or L2. Loop interchange improves the order of access within tiles. Packing copies non-contiguous slices into contiguous buffers, eliminating TLB misses and cache-line waste. Promotion materializes intermediate accumulation buffers in fast local memory.

**Transformations selected**:
- **Tiling**: The foundational transformation — partitions the 7D iteration space into blocks.
- **Loop Interchange**: Reorders loops within tiles to improve stride patterns (e.g., ensure stride-1 access is innermost).
- **Packing**: Copies operand tiles into contiguous, aligned buffers — critical for NCHW layout where channel strides are large.
- **Promotion**: Allocates local buffers for intermediate accumulation, reducing write-back traffic to main memory.

### Intent 2: SIMD Vectorization and Instruction-Level Parallelism (HIGH priority)

**Rationale**: AVX2 FMA instructions can execute 4 FP64 multiply-accumulates per cycle per core. Without vectorization, only scalar FMA is used — a 4x throughput loss. Vectorization requires an innermost loop with static bounds and stride-1 access. Padding ensures trip counts are multiples of the vector width. Unrolling exposes multiple independent vector operations to fill the FMA pipeline. Peeling separates the clean vectorized main loop from remainder iterations.

**Transformations selected**:
- **Vectorization**: Maps an inner loop dimension to SIMD vector operations.
- **Loop Unrolling**: Unrolls inner loops to expose ILP and fill FMA pipeline stages.
- **Padding**: Pads iteration dimensions to multiples of vector width for clean vectorized loops.
- **Peeling**: Splits a loop into a main body (with trip count divisible by tile/vector size) and a scalar remainder.

### Intent 3: Coarse-Grain Parallelism and Work Distribution (MEDIUM priority)

**Rationale**: With 28 cores available, parallelizing outer loops provides significant speedup for large workloads. The 4 parallel dimensions (batch, output-channel, output-height, output-width) offer ample parallelism. Fusion of producer-consumer chains reduces intermediate materialization. Loop distribution can isolate independent computations for better parallel scheduling.

**Transformations selected**:
- **Parallelization**: Distributes iterations of parallel loops across threads/cores via `scf.forall`.
- **Fusion**: Merges producer-consumer operations into shared tile loops, reducing memory traffic.
- **Loop Distribution**: Splits a single loop into multiple independent loops that can be scheduled or optimized separately.

### Intent 4: Iteration Space Transformation and IR Normalization (MEDIUM priority)

**Rationale**: Convolution has a specialized loop structure that can be lowered to a more regular matmul-like form via im2col, potentially unlocking more aggressive tiling and vectorization on the resulting contraction. Canonicalization simplifies the IR graph to remove redundancies and normalize patterns for downstream passes. Bufferization strategy controls how tensor semantics are lowered to buffer semantics, affecting allocation patterns and memory reuse.

**Transformations selected**:
- **Im2col Lowering**: Restructures convolution into an explicit data rearrangement followed by a matmul-like contraction — a well-known technique that can make the primary compute loop more regular.
- **Canonicalization**: Applies pattern-based simplifications and CSE to clean up the IR, often necessary between transformation steps.
- **Bufferization Strategy**: Controls the tensor-to-buffer lowering strategy (eliminate vs. explicit allocation), impacting memory allocation patterns and buffer reuse.

## Summary

| Intent | Priority | Transformations |
|--------|----------|----------------|
| Cache Locality and Data Reuse | HIGH | Tiling, Loop Interchange, Packing, Promotion |
| SIMD Vectorization and ILP | HIGH | Vectorization, Loop Unrolling, Padding, Peeling |
| Coarse-Grain Parallelism and Work Distribution | MEDIUM | Parallelization, Fusion, Loop Distribution |
| Iteration Space Transformation and IR Normalization | MEDIUM | Im2col Lowering, Canonicalization, Bufferization Strategy |

**Total**: 4 intents, 14 unique transformations — covering the full spectrum of CPU optimization opportunities for a convolution loop nest on Broadwell hardware.
