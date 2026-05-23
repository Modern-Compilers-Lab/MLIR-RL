# ReLU Action Enumeration Reasoning (v37)

## Workload Analysis

The input is a **ReLU activation** expressed as a `linalg.generic` with element-wise semantics:
- **All parallel iterators** — no reductions, no inter-element dependencies.
- **Trivial per-element compute**: one floating-point compare (`arith.cmpf ugt`) + one select (`arith.select`). This is ~2-3 FP/integer ops per element.
- **No data reuse**: each input element is read exactly once, each output element is written exactly once. There is no temporal locality to exploit.
- **Tensor semantics** (not buffer), with varying ranks: 2D (e.g., 128x1024) and 4D (e.g., 128x256x112x112).
- **f64 data type**: 8 bytes per element, which means AVX2 gives 4 lanes per vector register (256-bit).

## Shape Diversity

The 149 instances span a wide range of tensor sizes:
- **Small**: relu_256_256 (~65K elements, ~0.5 MB) — fits in L2/L3.
- **Medium**: relu_128_512_14_14 (~12.8M elements, ~100 MB) — exceeds per-core caches but fits in aggregate L3.
- **Large**: relu_128_256_112_112 (~411M elements, ~3.3 GB), relu_256_32_150_150 (~184M elements, ~1.5 GB) — far exceeds all cache levels.

This range means optimizations must be robust across sizes. Parallelization and vectorization are universally beneficial; tiling's value increases with tensor size (TLB pressure, NUMA effects).

## Performance Bottleneck: Memory Bandwidth

ReLU is **memory-bandwidth bound**. The arithmetic intensity is approximately:
- 2-3 ops per element / (2 * 8 bytes per element for read+write) ≈ 0.12-0.19 ops/byte

This is far below the machine's compute-to-bandwidth ratio. Therefore:
1. **Reducing instruction count** (vectorization) directly helps by processing 4 elements per instruction.
2. **Saturating memory bandwidth** requires using all 28 cores (parallelization).
3. **Cache-level optimizations** have limited direct benefit for standalone ReLU (no reuse), but help with TLB pressure on large tensors and create well-sized work units for parallelization.

## Intent Prioritization

### Intent 1: Data-Level Parallelism (SIMD) — HIGH
The compare+select pattern maps directly to AVX2 vector compare and blend instructions. Vectorizing the innermost loop reduces instruction count by 4x (for f64 on 256-bit AVX2). This is the single most impactful single-thread optimization.

Loop interchange is paired here because vectorization requires the target loop to have stride-1 (contiguous) memory access. While the default loop ordering for NCHW layout already has the innermost dimension contiguous, tiling may disrupt this, and 2D instances have different loop structures. Interchange ensures the correct loop is innermost for vectorization.

### Intent 2: Thread-Level Parallelism — HIGH
With all loops being embarrassingly parallel and large tensors spanning gigabytes, distributing work across 28 physical cores is essential to saturate memory bandwidth. Two flavors are proposed:
- **Tiling-based parallelization**: Tiles outer loops and maps tiles to threads. More flexible, handles arbitrary shapes, and composes well with subsequent tiling for cache blocking.
- **Num_threads-based parallelization**: Directly splits an iteration dimension across threads. Simpler but requires divisibility.

### Intent 3: Iteration Space Restructuring — MEDIUM
Tiling and unrolling restructure the loop nest for better hardware utilization:
- **Tiling** partitions the iteration space into blocks. For ReLU, the main benefit is (a) creating balanced parallel work units, (b) reducing TLB misses for very large tensors, and (c) enabling downstream vectorization of inner tiles.
- **Unrolling** reduces loop overhead (branch prediction, counter increment) and exposes instruction-level parallelism (ILP) to the out-of-order execution engine. For bandwidth-bound code, overlapping multiple independent load/store chains via unrolling can help hide memory latency.

## Transformations Not Included (and Why)

- **Fusion**: Not applicable — the RL system sees single operations, not producer-consumer pairs.
- **Promotion / Packing**: No benefit — ReLU has no data reuse across elements, so copying to contiguous buffers is pure overhead.
- **Peeling**: Useful as an internal implementation detail (handling vectorization remainders) but not a standalone RL action.
- **Im2col / Kernel-specific lowerings**: Not applicable to element-wise operations.
