# Action Enumeration Reasoning — v33 (dataset_conv2d, conv_2d_nchw_fchw)

## Operation Analysis

**Operation:** `linalg.conv_2d_nchw_fchw`
**Loop nest:** 7 dimensions — 4 parallel (N, F, OH, OW) × 3 reduction (C, KH, KW)
**Benchmark:** 278 instances, all `conv_2d_nchw_fchw` variants

### Dataset Shape Observations

| Feature | Range / Notes |
|---|---|
| Batch size (N) | 128, 256 |
| Input channels (C) | 32–288 |
| Output channels (F) | 32–384 |
| Spatial input (H×W) | 7×7 up to 28×28 |
| Kernel (KH×KW) | 1×1 (dominant) or 3×3 (minority) |
| Output spatial (OH×OW) | 4×4 up to 14×14 |
| Data type | f64 |

**Critical observation:** The majority of instances are **1×1 convolutions** (KH=KW=1). In this case:
- The reduction nest collapses to a single loop over C.
- The operation is structurally a batched matrix contraction: output[n,f,oh,ow] += input[n,c,oh,ow] * filter[f,c].
- Cache blocking and vectorization of C are the dominant levers.

For the **3×3 minority**: the KH×KW spatial reduction adds a small but non-trivial inner nest. Im2col lowering can convert this to a uniform matmul-like contraction.

---

## Hardware Target

- **CPU:** Intel Xeon E5-2680 v4 (Broadwell-class)
- **Cores:** 28 physical cores, no SMT
- **SIMD:** AVX2 (256-bit), no AVX-512; FP64: 4 lanes/vector
- **Cache:** L1d ~32KB/core, L2 ~256KB/core, shared L3

---

## Optimization Intent Selection

### Why 3 HIGH-priority intents?

For this operation on this hardware, there is no obvious "low-impact" class:

1. **Cache Locality** is essential because the input/filter working sets (N×C×H×W and F×C×KH×KW) far exceed per-core caches at full size. Without tiling, every convolution output pixel re-streams the full filter weight tensor.

2. **SIMD Vectorization** is essential because FP64 on AVX2 offers a 4× throughput multiplier; leaving vectorization unapplied means at best scalar execution with only 25% utilization of the available FP units.

3. **Thread Parallelism** is essential because 28 idle cores represent a 28× missed opportunity. The outer parallel loops (N×F) provide more than enough independent iterations for full utilization.

---

## Intent 1 — Cache Locality and Data Layout Regularization

### Transformations Selected

**Tiling:** The primary instrument for bounding the active working set. Blocking over (N, F, C) or (F, OH, OW, C) creates tiles that can reside in L1/L2 during inner computation. Tile sizes matching ~32KB (L1) or ~256KB (L2) are the natural choices for Layer 2 to parameterize.

**Loop Interchange:** Default MLIR lowering of conv_2d_nchw_fchw places loops in N-F-OH-OW-C-KH-KW order. For 1×1 kernels, moving C inward relative to OH/OW avoids re-loading filter weights per spatial step. For 3×3 kernels, placing KH-KW inside C amortizes input channel loads across the kernel window. The permutation parameter covers both scenarios.

**Packing:** After tiling, the sliced sub-tensors of the filter (F×C tile) or input (N×C×OH×OW tile) still carry the original NCHW strides. Packing rewrites them into contiguous buffers matching the tiled traversal order. This is a prerequisite for clean vectorization and removes gather-load penalties.

**Im2col Lowering:** For 3×3 convolutions, the KH×KW inner nest creates an irregular access pattern (sliding window) that resists simple vectorization. Im2col materializes the implicit sliding-window operand into an explicit 2D matrix (patches × channels), converting the entire operation into a standard matrix contraction. All downstream tiling/vectorization then targets the contraction op. Zero-parameter action — the transformation is either applied or not.

---

## Intent 2 — SIMD Vectorization and Local Buffer Exploitation

### Transformations Selected

**Vectorization:** After tiling and layout regularization, the innermost loops over output-width (OW tile) or input channels (C) are unit-stride and independent — ideal for AVX2 widening. At 4 FP64 lanes per vector, vectorization provides a direct 4× compute throughput improvement. The `tile_sizes` parameterization aligns vector tile sizes with the physical SIMD width.

**Promotion:** MLIR's vectorization pass requires stride-1 memref buffers. Tensor-level tiling produces sub-tensors that still carry full-tensor strides. Promotion inserts an explicit copy into a fresh, flat memref buffer (requiring internal bufferization), then canonicalization folds dynamic buffer shapes into static types. Without promotion, vectorization either fails or produces inefficient gather/scatter sequences. Promotion must target the outer tile (not the innermost loop) to amortize copy overhead.

---

## Intent 3 — Thread-Level Parallelism

### Transformations Selected

**Parallel Tiling (forall-based):** Creates independent work tiles over outer parallel loops using `scf.forall`, which maps directly to thread dispatch. The tile sizes parameterize the granularity of parallelism and interact with inner tiling choices, making this the more flexible and RL-friendly option. Compatible with inner sequential tiling and vectorization.

**Parallelization (num_threads):** Directly partitions the outermost loop into exactly `num_threads` parallel segments. Simpler scheduling, lower bookkeeping overhead, and a natural discrete choice for the RL agent when tile granularity is already set. The thread count must divide the iteration count; Layer 2 enforces this as a precondition.

Both are enumerated per the system guidance that explicitly calls for two parallelization variants.

---

## Transformations NOT Included

- **Unrolling:** Secondary effect on these loop sizes; register pressure is a concern at the given FP64 accumulator counts. Left to Layer 2/3 to evaluate as a follow-on.
- **Fusion:** No producer/consumer pair in the single-op template; not applicable at this level.
- **Bufferization (standalone):** Required internally by Promotion; not a standalone RL action for this op.
- **Canonicalization:** A step within Promotion, not a discrete optimization action for the RL agent.
