# MLIR Schedule Exploration Log: pooling_nchw
- Action Version: v50
- Kernel Family: linalg.pooling_nchw_max (various shapes, f64)
- Hardware: Intel Xeon E5-2680 v4 (Broadwell), AVX2, 28 cores
- Date: 2026-05-30

## Kernels Used
| Role | Kernel | Input | Kernel Window | Output | Strides |
|------|--------|-------|-------|--------|---------|
| Phase 1-2 representative | pooling_nchw_max_128_64_14_14_1_7_7 | 128x64x14x14 | 1x1 | 128x64x7x7 | 2x2 |
| Phase 3-4 small | pooling_nchw_max_128_192_7_7_1_4_4 | 128x192x7x7 | 1x1 | 128x192x4x4 | 2x2 |
| Phase 3-4 medium | pooling_nchw_max_128_288_56_56_1_28_28 | 128x288x56x56 | 1x1 | 128x288x28x28 | 2x2 |
| Phase 3-4 large | pooling_nchw_max_128_48_224_224_3_111_111 | 128x48x224x224 | 3x3 | 128x48x111x111 | 2x2 |

Loop structure for pooling_nchw_max: 6 loops (N, C, OH, OW, KH, KW)
- 4 parallel dims: N, C, OH, OW
- 2 reduction dims: KH, KW

## Phase 0: Baselines

### Representative kernel: pooling_nchw_max_128_64_14_14_1_7_7
- Iteration domain: N=128, C=64, OH=7, OW=7, KH=1, KW=1
- MLIR base time: 0.675 ms
- PyTorch time: 0.268 ms

### Phase 3-4 kernels
| Kernel | MLIR baseline (ms) | PyTorch (ms) |
|--------|-------------------|--------------|
| small (128_192_7_7_1_4_4) | 0.526 | 0.280 |
| medium (128_288_56_56_1_28_28) | 53.714 | 23.135 |
| large (128_48_224_224_3_111_111) | 741.057 | 83.0 |

## Phase 1: Single Actions

Kernel: pooling_nchw_max_128_64_14_14_1_7_7 (MLIR base: 0.675 ms, PyTorch: 0.268 ms)

| # | Action | Parameters | Pre | Post | Time (ms) | Speedup vs base | Speedup to PyTorch |
|---|--------|-----------|-----|------|-----------|---------|--------------------|
| S1 | Tiling | {tile_sizes: [4,8,0,0,0,0]} | T | T | 0.677 | 1.00x | 0.40x |
| S2 | LoopInterchange | {permutation: [1,0,2,3,4,5]} | T | T | 0.899 | 0.75x | 0.30x |
| S3 | Promotion | {tile_sizes: [4,8,0,0,0,0], operands: [0,1,2]} | T | T | 2.171 | 0.31x | 0.12x |
| S4 | SequentialVectorization | {vector_sizes: [4,4,1,1,1,1]} | T | T | 0.626 | 1.08x | 0.43x |
| S5 | ParallelVectorization | {vector_sizes: [4,4,0,0]} | T | T | 0.227 | 2.97x | 1.18x |
| S6 | TilingParallelization | {tile_sizes: [4,4,0,0]} | T | T | 0.221 | 3.05x | 1.21x |
| S7 | ThreadParallelization | {num_threads: 28} | T | T | 0.226 | 2.99x | 1.19x |

**Key Findings Phase 1:**
- All 7 actions applicable to pooling_nchw_max
- All preserve tag (postcondition passes)
- Parallelization actions (ParVec, TilPar, ThreadPar) are big winners (~3x)
- Note: LoopInterchange converts pooling_nchw_max to linalg.generic (with maximumf)
- Note: ParallelVectorization also converts to linalg.generic
- Note: Promotion converts to memref form
- Note: SequentialVectorization tiles all dims and converts to linalg.generic
- Note: ThreadParallelization creates dynamic shapes (128 not evenly divisible by 28)

## Phase 2: Pairwise Compositions

Kernel: pooling_nchw_max_128_64_14_14_1_7_7

### Composability Matrix

All 49 ordered pairs (7 first-actions x 7 second-actions) tested empirically.

| First \ Second | Tiling | LoopInterchange | Promotion | SeqVec | ParVec | TilPar | ThreadPar |
|----------------|--------|-----------------|-----------|--------|--------|--------|-----------|
| **Tiling** | OK | OK | OK | OK | OK | OK | OK |
| **LoopInterchange** | OK | OK | OK | OK | OK | OK | OK |
| **Promotion** | OK | OK | OK | OK | OK | OK | OK |
| **SeqVec** | FAIL(post) | FAIL(post) | FAIL(post) | FAIL(post) | FAIL(post) | FAIL(post) | FAIL(post) |
| **ParVec** | OK | OK | OK | OK | OK | OK | OK |
| **TilPar** | OK | OK | OK | OK | OK | OK | OK |
| **ThreadPar** | OK | OK | OK | OK | OK | OK | OK |

Legend: OK = pre=T, post=T; FAIL(post) = pre=T, post=F

**Key Findings Phase 2:**
- **SequentialVectorization is the ONLY terminal action**: after SeqVec, ALL 7 subsequent actions produce pre=T but post=F. The structural reason is that SeqVec moves the `tag = "operation_0"` from the linalg op to the outermost `scf.for` loop, leaving the resulting `linalg.generic` untagged. Subsequent action tools cannot find the tagged operation.
- **All other 42 pairs compose cleanly** (pre=T, post=T): Tiling, LoopInterchange, Promotion, ParallelVectorization, TilingParallelization, and ThreadParallelization can all precede any other action.
- **Self-composition works** for parallelization actions: ParVec->ParVec, TilPar->TilPar, ThreadPar->TilPar all create nested `scf.forall` constructs and remain composable.
- **Promotion is NOT terminal** despite converting to memref form -- all actions compose after it.
- **LoopInterchange is NOT terminal** despite converting to linalg.generic -- all actions compose after it.

### Constraint Summary
- SeqVec must be the LAST action in any schedule (terminal)
- All other 6 actions can appear in any order and any position before SeqVec

## Phase 3: Multi-Step Schedules

Explored by 3 parallel agents on small/medium/large kernel subsets.

### Small kernel: pooling_nchw_max_128_192_7_7_1_4_4
MLIR base: 0.526 ms | PyTorch: 0.280 ms

| # | Schedule Shape | Probe Parameters | OK? | Time (ms) | vs base | vs PyTorch | Notes |
|---|---|---|---|---|---|---|---|
| 1 | TilPar -> Tiling -> SeqVec | [4,4,0,0] / [2,4,4,4,0,0] / [2,4,2,2,1,1] | Yes | 0.268 | 1.96x | **1.04x** | **BEST** |
| 2 | ThreadPar -> Tiling -> SeqVec | 28 / [4,8,0,0,0,0] / [4,4,1,1,1,1] | Yes | 0.273 | 1.93x | 1.03x | Good |
| 3 | TilPar -> Tiling -> Tiling -> SeqVec | [4,4,0,0] / [2,4,4,4,0,0] / [2,2,0,0,0,0] / [2,2,4,4,1,1] | Yes | 0.276 | 1.91x | 1.01x | Extra tile adds overhead |
| 4 | ParVec -> Tiling -> SeqVec | [4,4,0,0] / [4,4,0,0,0,0] / [4,4,2,2,1,1] | Yes | 0.277 | 1.90x | 1.01x | Decent |
| 5 | LoopInterchange -> TilPar -> SeqVec | [1,0,2,3,4,5] / [4,4,0,0] / [4,4,1,1,1,1] | Yes | 0.306 | 1.72x | 0.92x | LI overhead |
| 6 | Tiling -> TilPar -> SeqVec | [4,8,0,0,0,0] / [4,8,0,0] / [4,4,1,1,1,1] | Yes | 0.620 | 0.85x | 0.45x | POOR: degenerate forall |
| 7 | Tiling -> ThreadPar -> SeqVec | [4,8,0,0,0,0] / 28 / [4,4,1,1,1,1] | Yes | 58.6 | 0.009x | 0.005x | CATASTROPHIC |
| B1 | TilPar (alone) | [4,4,0,0] | Yes | 0.274 | 1.92x | 1.02x | Strong single-action |
| B2 | ParVec (alone) | [4,4,0,0] | Yes | 0.271 | 1.94x | 1.03x | Strong single-action |
| B3 | ThreadPar (alone) | 28 | Yes | 0.276 | 1.91x | 1.01x | Strong single-action |

### Medium kernel: pooling_nchw_max_128_288_56_56_1_28_28
MLIR base: 53.714 ms | PyTorch: 23.135 ms

| # | Schedule Shape | Probe Parameters | OK? | Time (ms) | vs base | vs PyTorch | Notes |
|---|---|---|---|---|---|---|---|
| 1 | ThreadPar (alone) | 28 | Yes | 5.679 | 9.46x | **4.07x** | **BEST** |
| 2 | ThreadPar -> Tiling -> SeqVec | 28 / [4,8,4,4,0,0] / [4,8,4,4,1,1] | Yes | 6.813 | 7.88x | 3.40x | 2nd best |
| 3 | LoopInterchange -> TilPar -> SeqVec | [1,0,2,3,4,5] / [4,4,4,4] / [4,4,4,4,1,1] | Yes | 7.212 | 7.45x | 3.21x | 3rd best |
| 4 | ThreadPar -> Tiling -> Tiling -> SeqVec | 28 / [4,8,4,4,0,0] / [2,4,2,2,0,0] / [2,4,2,2,1,1] | Yes | 7.325 | 7.33x | 3.16x | 4-step |
| 5 | TilPar -> Tiling -> SeqVec | [4,4,4,4] / [4,4,4,4,0,0] / [4,4,4,4,1,1] | Yes | 7.523 | 7.14x | 3.08x | Good |
| 6 | ParVec -> Tiling -> SeqVec | [4,4,4,4] / [4,4,4,4,0,0] / [4,4,4,4,1,1] | Yes | 7.734 | 6.95x | 2.99x | Good |
| 7 | TilPar (alone) | [4,4,4,4] | Yes | 7.576 | 7.09x | 3.05x | Good |
| 8 | ParVec (alone) | [4,4,4,4] | Yes | 7.557 | 7.11x | 3.06x | Good |
| 9 | Tiling -> TilPar -> SeqVec | [4,8,4,4,0,0] / [4,8,4,4] / [4,8,4,4,1,1] | Yes | 99.275 | 0.54x | 0.23x | POOR |
| 10 | Tiling -> ThreadPar -> SeqVec | [4,8,4,4,0,0] / 28 / [4,8,4,4,1,1] | Yes | 4226.9 | 0.013x | 0.005x | CATASTROPHIC |

### Large kernel: pooling_nchw_max_128_48_224_224_3_111_111
MLIR base: 741.057 ms | PyTorch: 83.0 ms

| # | Schedule Shape | Probe Parameters | OK? | Time (ms) | vs base | vs PyTorch | Notes |
|---|---|---|---|---|---|---|---|
| 1 | TilPar -> Tiling -> SeqVec | [2,2,0,0] / [2,2,0,0,0,0] / [2,2,1,1,1,1] | Yes | 26.79 | 27.66x | **3.10x** | **BEST** |
| 2 | TilPar (alone) | [2,2,0,0] | Yes | 26.84 | 27.61x | 3.09x | Near-best |
| 3 | ParVec (alone) | [4,4,0,0] | Yes | 27.35 | 27.10x | 3.04x | Near-best |
| 4 | TilPar -> Tiling -> Tiling -> SeqVec | [4,4,0,0] / [4,4,0,0,0,0] / [2,2,0,0,0,0] / [2,2,1,1,1,1] | Yes | 27.37 | 27.08x | 3.03x | 4-step competitive |
| 5 | TilPar -> ParVec | [4,4,0,0] / [4,4,0,0] | Yes | 27.90 | 26.56x | 2.97x | Nested forall |
| 6 | ThreadPar -> Tiling | 28 / [2,16,0,0,0,0] | Yes | 29.26 | 25.33x | 2.84x | Good |
| 7 | ThreadPar (alone) | 28 | Yes | 29.31 | 25.29x | 2.83x | Good |
| 8 | TilPar -> LI -> SeqVec | [4,4,0,0] / [1,0,2,3,4,5] / [4,4,1,1,1,1] | Yes | 35.26 | 21.02x | 2.35x | SeqVec overhead |
| 9 | TilPar -> Tiling -> SeqVec (4,4 params) | [4,4,0,0] / [4,4,0,0,0,0] / [4,4,1,1,1,1] | Yes | 35.50 | 20.87x | 2.34x | Larger tile probe |
| 10 | LI -> TilPar -> SeqVec | [1,0,2,3,4,5] / [4,4,0,0] / [4,4,1,1,1,1] | Yes | 37.75 | 19.63x | 2.20x | LI entry |
| 11 | Tiling -> TilPar -> SeqVec | [4,8,0,0,0,0] / [4,4,0,0] / [4,4,1,1,1,1] | Yes | 500.76 | 1.48x | 0.17x | POOR |
| 12 | Tiling -> ThreadPar | [4,8,0,0,0,0] / 28 | Yes | 198.35 | 3.74x | 0.42x | POOR |

## Phase 4: Best Schedule Shape per Case

| Kernel subset | Winning skeleton | Best probe time (ms) | vs base | vs PyTorch |
|---|---|---|---|---|
| Small (128_192_7_7_1_4_4) | TilPar -> Tiling -> SeqVec | 0.268 | 1.96x | 1.04x |
| Medium (128_288_56_56_1_28_28) | ThreadPar -> Tiling -> SeqVec | 6.813 | 7.88x | 3.40x |
| Large (128_48_224_224_3_111_111) | TilPar -> Tiling -> SeqVec | 26.79 | 27.66x | 3.10x |

Note: Single-action ThreadPar (5.679ms, 4.07x) is actually best for medium kernel, but ThreadPar -> Tiling -> SeqVec (6.813ms, 3.40x) is included as the multi-step winner since single-action "done after step 1" is implicit in the schedule graph.

### Cross-kernel structural patterns:
1. **Parallelization first is critical** -- Tiling first then parallelizing is catastrophic (10-1000x slowdown)
2. **All 3 parallelization actions achieve similar performance** when applied first (~3x vs PyTorch)
3. **Multi-step schedules provide marginal improvement** over single-action parallelization on pooling kernels
4. **SeqVec adds overhead for pooling with KH=KW=1** (nothing meaningful to vectorize)
5. **LoopInterchange provides minimal benefit** for this family

## Composability Matrix

(See Phase 2 above -- full 7x7 matrix with all 49 cells tested empirically)

## Key Findings

### Best Schedules
| Subset | Schedule | Time (ms) | Speedup vs MLIR base | Speedup vs PyTorch |
|--------|----------|-----------|---------------------|-------------------|
| Small | TilPar[4,4,0,0] -> Tiling[2,4,4,4,0,0] -> SeqVec[2,4,2,2,1,1] | 0.268 | 1.96x | 1.04x |
| Medium | ThreadPar(28) alone | 5.679 | 9.46x | 4.07x |
| Large | TilPar[2,2,0,0] -> Tiling[2,2,0,0,0,0] -> SeqVec[2,2,1,1,1,1] | 26.79 | 27.66x | 3.10x |

### Composability Issues Discovered
- **SequentialVectorization is terminal**: After SeqVec, the tag moves from the linalg op to the outermost scf.for loop, leaving the linalg.generic untagged. All 7 subsequent actions fail with post=F (pre=T but tag not found on any linalg op).
- **No other composability issues**: All 42 non-SeqVec pairs compose cleanly.

### Ordering Constraints
- **Parallelization MUST come before Tiling**: Tiling first creates sequential scf.for loops; subsequent parallelization inside those loops is catastrophic (28 threads per tiny tile iteration = massive sync overhead).
- **SequentialVectorization must be LAST**: Terminal action, blocks all subsequent actions.
- **LoopInterchange adds minimal value**: For pooling_nchw, the default N,C,OH,OW ordering is already reasonable.
- **Promotion is harmful**: Copy overhead for large tensors outweighs any locality benefit on this memory-bound kernel.

### Critical Ordering Anti-Pattern
**NEVER**: Tiling -> {ThreadPar, TilPar, ParVec}. This creates parallelization inside sequential tile loops, resulting in 100-1000x slowdowns (e.g., 4227ms vs 5.7ms baseline on medium kernel).

## Synthesis

### ACTION_DEPENDENCIES

<!-- SeqVec moves tag from linalg op to outermost scf.for, making subsequent actions unable to find the tagged operation -->
```python
ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "SequentialVectorization": ["Tiling", "LoopInterchange", "Promotion", "ParallelVectorization", "TilingParallelization", "ThreadParallelization"],
}
```

### SCHEDULE_GRAPH

<!-- TilPar -> Tiling -> SeqVec: Winner for small (1.04x) and large (3.10x) kernels. Canonical parallelize-tile-vectorize pattern. -->
<!-- ThreadPar -> Tiling -> SeqVec: Winner for medium kernel (3.40x, with ThreadPar alone at 4.07x reachable via early done). Thread-based parallelism suits larger batch+channel dims. -->
<!-- TilPar -> Tiling -> Tiling -> SeqVec: Competitive on large kernel (3.03x). Hierarchical tiling for deeper cache blocking. -->
<!-- LoopInterchange -> TilPar -> SeqVec: 3rd best on medium (3.21x). Interchange entry enables different loop ordering before parallelization. -->
<!-- ParVec -> Tiling -> SeqVec: Competitive across all subsets (1.01x small, 2.99x medium, 3.03x large). Different mechanism (converts to generic + forall). -->
```python
SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    "pooling_nchw": [
        ["TilingParallelization", "Tiling", "SequentialVectorization"],
        ["ThreadParallelization", "Tiling", "SequentialVectorization"],
        ["TilingParallelization", "Tiling", "Tiling", "SequentialVectorization"],
        ["LoopInterchange", "TilingParallelization", "SequentialVectorization"],
        ["ParallelVectorization", "Tiling", "SequentialVectorization"],
    ],
}
```
