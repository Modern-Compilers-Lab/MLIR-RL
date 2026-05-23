# MLIR Schedule Exploration Log: pooling_nchw_max_256_128_28_28_1_14_14
- Action Version: v34
- Kernel: linalg.pooling_nchw_max 256x128x28x28 input, 1x1 kernel, 256x128x14x14 output, stride=2, f64
- Hardware: Intel Xeon E5-2680 v4 (Broadwell), AVX2, 28 cores
- Date: 2026-05-20

## Dimension Analysis
- Loop dimensions: [N=256, C=128, OH=14, OW=14, KH=1, KW=1] — 6 loops
- Tiling vocab [0,4,8,16,32]: N(256) -> {4,8,16,32}; C(128) -> {4,8,16,32}; OH(14) -> {0 only}; OW(14) -> {0 only}; KH(1) -> {0}; KW(1) -> {0}
- Promotion vocab [0,16,32,64,128]: N(256) -> {16,32,64,128}; C(128) -> {16,32,64,128}; OH(14) -> {0}; OW(14) -> {0}; KH(1) -> {0}; KW(1) -> {0}
- Vectorization vocab [4,8,16,32] (all > 0): OH=14, OW=14 not divisible by any; KH=1, KW=1 not divisible by any. Vectorization structurally impossible for this kernel.
- Parallelization vocab [0,32,64,128,256]: N(256) -> {32,64,128,256}; C(128) -> {32,64,128}; rest -> {0}
- ParallelizationByNumThreads vocab [2,4,8,14,28]: all valid

## Baseline
- MLIR base time: 12.40 ms
- PyTorch time: 5.60 ms

## Phase 1: Single Actions
| # | Action | Parameters | Pre | Post | Time (ms) | Speedup | Speedup to PyTorch |
|---|--------|-----------|-----|------|-----------|---------|--------------------|
| 1 | Tiling | [4,4,0,0,0,0] | T | T | 12.47 | 0.99x | 0.45x |
| 2 | Tiling | [16,16,0,0,0,0] | T | T | 12.44 | 1.00x | 0.45x |
| 3 | Tiling | [32,32,0,0,0,0] | T | T | 12.57 | 0.99x | 0.45x |
| 4 | LoopInterchange | [2,3,0,1,4,5] | T | T | 41.10 | 0.30x | 0.14x |
| 5 | LoopInterchange | [1,0,2,3,4,5] | T | T | 13.59 | 0.91x | 0.41x |
| 6 | LoopInterchange | [0,1,3,2,4,5] | T | T | 14.05 | 0.88x | 0.40x |
| 7 | Promotion | [16,16,0,0,0,0] | T | T | 56.83 | 0.22x | 0.10x |
| 8 | Promotion | [64,64,0,0,0,0] | T | T | 65.89 | 0.19x | 0.09x |
| 9 | Vectorization | [4,4,4,4,4,4] | F | F | FAIL | - | - |
| 10 | Unrolling | dim=0, f=4 | T | T | 12.54 | 0.99x | 0.45x |
| 11 | Unrolling | dim=1, f=4 | T | T | 13.52 | 0.92x | 0.41x |
| 12 | Unrolling | dim=2, f=2 | T | T | 25.01 | 0.50x | 0.22x |
| 13 | Parallelization | [32,0,0,0,0,0] | T | T | 2.47 | 5.02x | 2.27x |
| 14 | Parallelization | [64,64,0,0,0,0] | T | T | 2.46 | 5.04x | 2.28x |
| 15 | Parallelization | [128,128,0,0,0,0] | T | T | 6.73 | 1.84x | 0.83x |
| 16 | **Parallelization** | **[32,32,0,0,0,0]** | T | T | **1.76** | **7.05x** | **3.18x** |
| 17 | ParallelByThreads | 4 | T | T | 3.47 | 3.57x | 1.61x |
| 18 | ParallelByThreads | 14 | T | T | 2.23 | 5.56x | 2.51x |
| 19 | **ParallelByThreads** | **28** | T | T | **1.64** | **7.54x** | **3.41x** |

### Phase 1 Observations
- **Parallelization dominates**: This is a memory-bound kernel where work is embarrassingly parallel over N and C. Using all 28 cores yields 7.54x speedup.
- **Tiling/Unrolling/LoopInterchange provide no benefit**: Single-core optimizations don't help when the bottleneck is memory bandwidth, not compute.
- **Promotion is harmful**: Data copies to/from alloca buffers add overhead (4-5x slowdown) with no cache benefit since the kernel is already streaming through memory.
- **Vectorization is structurally impossible**: OH=14, OW=14, KH=1, KW=1 — none divisible by any vectorization vocab value {4,8,16,32}.
- **Parallelization with too many tiles hurts**: Pa[128,128] creates 2x1=2 threads (insufficient), while Pa[32,32] creates 8x4=32 threads (near-optimal for 28 cores).

## Phase 2: Composability Matrix

Tested ALL 7x7 = 49 pairwise compositions empirically. Vectorization was tested as A (standalone) and as B (after every A).

| A \ B | Tiling | LoopInterchange | Promotion | Vectorization | Unrolling | Parallelization | PBT |
|-------|--------|----------------|-----------|---------------|-----------|----------------|-----|
| **Tiling** | OK | OK | FAIL* | FAIL | OK | OK(trivial) | OK |
| **LoopInterchange** | OK | OK | OK | FAIL | OK | OK | OK |
| **Promotion** | OK | OK | OK | FAIL | OK | OK(trivial) | OK |
| **Vectorization** | - | - | - | - | - | - | - |
| **Unrolling** | OK | OK | OK | FAIL | OK | OK(trivial) | OK(trivial) |
| **Parallelization** | OK | OK | OK | FAIL | OK | OK(trivial) | OK |
| **PBT** | OK | OK | OK | FAIL | OK | OK | OK |

Legend: OK = pre=T, post=T; FAIL = pre=F; OK(trivial) = structurally valid but creates 1 thread (no-op parallelization); - = never applies as A

*T[4,4,0,0,0,0]->P fails because inner dims [1-4,4,14,14,1,1] have no valid promotion tile from vocab {16,32,64,128}. Larger T params (e.g., T[16,16]) can compose with P.

### Composability Findings
1. **Vectorization ALWAYS fails as B** (pre=F) regardless of A — structural due to kernel dimensions (OH=14, KH=KW=1), not due to predecessor action.
2. **Vectorization never applies as A** — same structural reason. Vectorization is completely inapplicable for this kernel shape.
3. **Promotion is NOT terminal** — all actions compose after Promotion (T, LI, P, U, Pa, PBT all OK after P). This contradicts any assumption that Promotion blocks follow-up actions.
4. **All non-Vectorization actions compose freely** with each other. The only failures are Vectorization-related.
5. **Parallelization after Tiling/Unrolling creates trivial parallelism** (1 thread) because N is already tiled to small values — structurally valid but useless.

## Phase 3: Multi-Step Schedules

| # | Schedule | Time (ms) | Speedup | Speedup to PyTorch |
|---|----------|-----------|---------|---------------------|
| 1 | PBT(28) | 1.64 | 7.54x | 3.41x |
| 2 | PBT(28) -> Tiling[0,4,0,0,0,0] | 1.61 | 7.70x | 3.48x |
| 3 | PBT(28) -> Tiling[0,8,0,0,0,0] | 1.61 | 7.70x | 3.48x |
| 4 | PBT(28) -> Tiling[0,16,0,0,0,0] | 1.62 | 7.65x | 3.46x |
| 5 | Pa[32,32,0,0,0,0] | 1.76 | 7.05x | 3.18x |
| 6 | Pa[32,32,0,0,0,0] -> Tiling[4,4,0,0,0,0] | 1.79 | 6.93x | 3.13x |
| 7 | PBT(28) -> Promotion[16,16,0,0,0,0] | 6.34 | 1.96x | 0.88x |

### Phase 3 Observations
- Multi-step schedules provide negligible improvement over PBT(28) alone. C-tiling within parallel chunks shows ~2% improvement (within noise).
- Promotion after parallelization destroys the parallelism benefit through copy overhead.
- The kernel is fundamentally memory-bandwidth-limited; once fully parallelized, no further optimization helps meaningfully.

## Phase 4: Local Tuning

Best schedule: **ParallelizationByNumThreads(28)** optionally followed by Tiling[0,4,0,0,0,0]

Tuning PBT thread count:
| Threads | Time (ms) | Speedup |
|---------|-----------|---------|
| 4 | 3.47 | 3.57x |
| 14 | 2.23 | 5.56x |
| 28 | 1.64 | 7.54x |

Tuning C-tile size after PBT(28):
| C-tile | Time (ms) | Speedup |
|--------|-----------|---------|
| 4 | 1.61 | 7.70x |
| 8 | 1.61 | 7.70x |
| 16 | 1.62 | 7.65x |
| None | 1.64 | 7.54x |

All C-tile sizes perform identically — the improvement is within measurement noise.

## Final Results
- **Best schedule**: PBT(28) + Tiling[0,4,0,0,0,0] (or PBT(28) alone)
- **Best time**: 1.61 ms (vs base 12.40 ms, PyTorch 5.60 ms)
- **Speedup vs MLIR base**: 7.70x
- **Speedup vs PyTorch**: 3.48x

## ACTION_DEPENDENCIES (pooling_nchw_max)

Based on exhaustive empirical testing of all 49 pairwise compositions:

```python
ACTION_DEPENDENCIES: dict[str, list[str]] = {
    # Vectorization never applies for this kernel shape (OH=14, KH=KW=1 not divisible by vocab).
    # When it does apply (on other kernels), it is terminal — blocks all follow-up actions.
    "Vectorization": ["Tiling", "LoopInterchange", "Promotion", "Vectorization", "Unrolling", "Parallelization", "ParallelizationByNumThreads"],
    # Promotion does NOT block any action — all 6 non-Vec actions compose after Promotion.
    # Previous assumption that Promotion blocks Vectorization is CONFIRMED (Vec fails after Promotion),
    # but this is due to Vec's own structural constraints, not Promotion creating a block.
}
```

### Key Dependency Insights
1. **Vectorization is terminal** (when applicable): blocks all 7 actions including itself. Confirmed from prior kernel explorations, not directly testable here.
2. **Promotion is NOT terminal**: contradicts the previous `"Promotion": ["Vectorization"]` entry. Vec fails after Promotion, but Vec fails after EVERYTHING for this kernel — the block edge is on Vectorization's preconditions, not on Promotion's output.
3. **No other action creates block edges**: T, LI, U, Pa, PBT all compose freely after each other and after themselves.
4. **Recommendation**: Remove the `"Promotion": ["Vectorization"]` edge from ACTION_DEPENDENCIES, as it's not a genuine dependency — Vectorization's own precondition checking handles this case.
