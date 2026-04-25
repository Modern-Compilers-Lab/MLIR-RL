# Layer-1 Action Enumeration — Reasoning

## Input under analysis

The sample payload is a structured `linalg` operation on dense floating-point tensors:

- `linalg.conv_2d_nchw_fchw` with a concrete instance of shape `128x32x7x7` (input) · `256x32x1x1` (filter) → `128x256x7x7` (output), dilation 1, stride 1.
- The broader system also targets matmul / contraction-shaped payloads that share the same class of loop nest.

Both families are regular, reduction-bearing loop nests with predictable iteration spaces, large per-op working sets, and abundant data reuse. I therefore reason from a **generic loop-nest viewpoint** — parallel loops, reduction loops, stride patterns, per-tile footprints, producer-consumer chains — rather than from kernel-specific dimension roles (no batch/channel/spatial/filter references in action templates).

## Target hardware profile (guides every priority)

- Intel Xeon E5-2680 v4 (Broadwell), 2 sockets × 14 cores = **28 physical cores**, SMT off, 2 NUMA nodes.
- AVX2 + FMA, **no AVX-512**: FP64 → 4-lane, FP32 → 8-lane 256-bit SIMD.
- Per-core L1d ≈ 32 KB, L2 ≈ 256 KB, shared L3 in the tens of MB per socket.

Three first-order performance levers dominate this target for dense loop-nest ML kernels:

1. **Cache and register reuse** — working sets must be blocked so the hot portion fits in L1/L2, otherwise the kernel collapses to DRAM bandwidth.
2. **SIMD utilization** — AVX2 + FMA gives ~4–8× peak FLOPs only if the innermost loop is vectorizable and has clean, divisible trip counts.
3. **Thread-level parallelism** — all 28 cores must contribute, with attention to per-core footprint, NUMA, and avoiding oversubscription.

Everything else is either an enabler of these three levers or a secondary concern.

## How the intents were chosen

I grouped transformations by the *performance lever* they primarily serve, so that a hierarchical RL policy can first pick the *intent* (the "why") and then the concrete *action* (the "what"):

1. **Memory Hierarchy Exploitation** — HIGH
2. **SIMD Vectorization and Instruction Throughput** — HIGH
3. **Coarse-Grain Parallelism and Work Distribution** — HIGH
4. **Iteration Space Normalization and Structured Lowering** — MEDIUM

Intents 1–3 correspond one-to-one with the three first-order performance levers and are all HIGH priority. Intent 4 is an enabler — it does not directly produce speedup, but it unlocks and multiplies the effectiveness of the first three intents, so it is MEDIUM.

## How the transformations were chosen per intent

### 1. Memory Hierarchy Exploitation (HIGH)

- **Tiling** — the foundational cache-blocking knob; required to keep large matmul/conv working sets in L1/L2. Parameterized over a loop band with per-loop tile sizes so multi-level (L3 → L2 → L1 → register) tiling is expressible by chaining applications or nested tile vectors in Layer 2.
- **LoopInterchange** — decides the stride pattern of the innermost loop and which operand stays resident in registers across the inner loop. Essential to make tiled loops actually benefit from the cache.
- **Packing** — materializes contiguous, optionally re-blocked operand panels to remove gather-like access patterns, mirroring the A/B panel strategy used by tuned BLAS GEMM and convolution-as-contraction pipelines.
- **Promotion** — copies a small sub-tile into a freshly allocated local buffer so the inner loop works on a private, fast copy. Kept distinct from Packing because it is a *localization* step without a layout rewrite; RL should be able to pick one or the other (or both).

### 2. SIMD Vectorization and Instruction Throughput (HIGH)

- **Vectorization** — the direct conversion of scalar loop iterations to SIMD instructions; indispensable to reach AVX2/FMA peak throughput.
- **LoopUnrolling** — exposes instruction-level parallelism, enables multiple independent FMA accumulators to hide latency, and increases register reuse across replicated iterations.
- **Peeling** — splits off ragged boundary iterations so the hot main loop has a clean, uniform trip count suitable for vectorization and unrolling.
- **Padding** — the complement of Peeling: removes the ragged edges up-front by extending operand dimensions or iteration bounds with neutral values, so tile sizes and vector widths divide evenly.

Peeling and Padding are kept as **two separate macro actions** because they solve the same "ragged trip count" problem via opposite mechanisms (split vs. extend). The RL agent should be able to choose between them per situation.

### 3. Coarse-Grain Parallelism and Work Distribution (HIGH)

- **Parallelization** — the action that actually turns a parallel loop band into multi-threaded execution across cores.
- **Fusion** — merges producer/consumer loop nests to increase per-task arithmetic intensity and eliminate intermediate materialization; particularly important when the payload contains fusible pre/post-processing around the main contraction.
- **LoopDistribution** — the inverse of Fusion: splits a loop nest so that one portion (e.g., a parallel part) can be parallelized/vectorized while another (e.g., a reduction) is handled separately or left scalar.

Both Fusion and LoopDistribution live in this intent because they shape the *granularity* of the work that parallel threads receive, which is the key quality for the parallel lever on this hardware.

### 4. Iteration Space Normalization and Structured Lowering (MEDIUM)

- **Canonicalization** — removes IR noise (dead code, trivial folds, redundant ops) that would otherwise block pattern matches used by later transformations. Used as a cheap step between heavier optimizations.
- **Im2colLowering** — explicitly permitted by the system description as an acceptable kernel-specific transformation when framed generically. It lets convolution-shaped payloads reuse the full mature matmul recipe (Tiling / Packing / Vectorization / Parallelization), which is often the fastest path on CPU.
- **BufferizationStrategy** — controls whether tensor→buffer lowering materializes large temporaries or performs in-place updates, with large downstream effects on memory bandwidth, layout stability, and the legality of later fusion/promotion.

## Deliberately excluded (and why)

- **Unroll-and-jam as a single macro action** — expressible as a composition of Tiling + LoopUnrolling (+ optional LoopInterchange). The system instructions explicitly warn against compound macro actions where the components can stand alone, so it is split.
- **Loop Collapse / coalescing** — niche on CPU; in practice subsumed by Tiling and LoopInterchange for this workload class.
- **Software pipelining, register allocation hints** — too microarchitecture-specific and poorly supported in the MLIR Transform dialect for this target.
- **GPU-only transformations** (warp/block tiling, shared memory) — explicitly out of scope per the system description.
- **Algorithmic rewrites** (Strassen, Winograd, FFT-based conv) — change numerical semantics and are out of scope by definition.
- **Per-dimension variants** (e.g., "Tile Output Spatial Dimensions", "Tile Input Channel") — forbidden by the Layer-1 granularity rules. These differences are pure parameterization and belong in Layer 2.

## Adherence to Layer-1 boundaries

This artifact (JSON + reasoning) is strictly a **catalog of candidate macro RL actions with rationale**. Within Layer 1 I did **not**:

- write any MLIR Transform dialect code,
- choose numeric parameter ranges or defaults,
- specify preconditions, postconditions, legality rules, or ordering constraints,
- describe or debug implementation details,
- validate action composition or execution.

Those responsibilities are deferred to Layer 2 (executable action synthesis) and Layer 3 (schedule verification and benchmarking), as prescribed by the system description.
