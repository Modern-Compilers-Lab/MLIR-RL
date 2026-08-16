# MLIR Schedule Exploration Log: All Kernel Families
- Action Version: v53 (tools: v55, includes Packing and Unrolling)
- Kernels: matmul_512_256_1536, conv2d_128_128_14_14_192_1_1_7_7, pooling_128_128_56_56_1_28_28, add_112_112_120_150, relu_128_128_56_56
- Hardware: Intel Xeon E5-2680 v4 (Broadwell), AVX2, 28 cores
- Date: 2026-08-15

## Baselines

| Family | Kernel | MLIR Base (ms) | PyTorch (ms) |
|--------|--------|---------------|-------------|
| matmul | 512x256 @ 256x1536 | 477.88 | 1.17 |
| conv2d | 128x128x14x14 * 192x128x1x1 → 128x192x7x7 | 174.15 | 0.87 |
| pooling | 128x128x56x56, k=1x1, out=28x28 | 23.63 | 11.15 |
| add | 112x112x120x150 | 338.45 | 68.09 |
| relu | 128x128x56x56 | 65.37 | 12.22 |

## Phase 1: Single Actions

### Matmul (512x256 @ 256x1536, base=477.88ms, torch=1.17ms)

| # | Action | Parameters | Pre | Post | Time (ms) | Speedup |
|---|--------|-----------|-----|------|-----------|---------|
| S1 | Tiling | {tile_sizes: [64,64,64]} | T | T | 331.16 | 1.44x |
| S2 | LoopInterchange | {permutation: [1,0,2]} | T | T | 542.38 | 0.88x |
| S3 | Promotion | {operands: [0,1,2]} | T | T | 262.79 | 1.82x |
| S4 | SeqVec | {vector_sizes: [4,4,4]} | T | T | 45.98 | 10.39x |
| S5 | ParVec | {vector_sizes: [4,4,4]} | T | T | 8.47 | 56.42x |
| S6 | Im2colLowering | {} | F | F | N/A | N/A |
| S7 | TilingPar | {tile_sizes: [64,64,0]} | T | T | 19.45 | 24.57x |
| S8 | ThreadPar | {num_threads: 28} | T | T | 19.75 | 24.19x |
| S9 | Packing | {packed_sizes: [32,32,32]} | T | T | 94.40 | 5.06x |
| S10 | Unrolling | {unroll_factor: 4} | T | T | 482.18 | 0.99x |

**Applicable:** Tiling, LoopInterchange, Promotion, SeqVec, ParVec, TilingPar, ThreadPar, Packing, Unrolling (9/10)

### Conv2d (128x128x14x14 * 192x128x1x1, base=174.15ms, torch=0.87ms)

| # | Action | Parameters | Pre | Post | Time (ms) | Speedup |
|---|--------|-----------|-----|------|-----------|---------|
| S1 | Tiling | {tile_sizes: [16,16,0,0,16,0,0]} | T | T | 130.30 | 1.34x |
| S2 | LoopInterchange | {permutation: [1,0,2,3,4,5,6]} | T | T | (not exec) | - |
| S3 | Promotion | {operands: [0,1,2]} | T | T | (not exec) | - |
| S4 | SeqVec | {vector_sizes: [16,16,1,1,16,1,1]} | F | F | N/A | N/A |
| S5 | ParVec | {vector_sizes: [16,16,1,1,16,1,1]} | F | F | N/A | N/A |
| S6 | Im2colLowering | {} | T | T | 178.88 | 0.97x |
| S7 | TilingPar | {tile_sizes: [16,16,7,7,0,0,0]} | T | T | 7.46 | 23.34x |
| S8 | ThreadPar | {num_threads: 28} | T | T | 6.91 | 25.20x |
| S9 | Packing | {packed_sizes: [16,16,0,0,16,0,0]} | T | T | (not exec) | - |
| S10 | Unrolling | {unroll_factor: 4} | T | T | (not exec) | - |

**Applicable:** Tiling, LoopInterchange, Promotion, Im2colLowering, TilingPar, ThreadPar, Packing, Unrolling (8/10)
**Not applicable:** SeqVec (pre=F), ParVec (pre=F)

### Pooling (128x128x56x56, k=1x1, out=28x28, base=23.63ms, torch=11.15ms)

| # | Action | Parameters | Pre | Post | Time (ms) | Speedup |
|---|--------|-----------|-----|------|-----------|---------|
| S1 | Tiling | {tile_sizes: [16,16,4,4,0,0]} | T | T | (pending) | - |
| S2 | LoopInterchange | {permutation: [1,0,2,3,4,5]} | T | T | (pending) | - |
| S3 | Promotion | {operands: [0,1,2]} | T | T | (pending) | - |
| S4 | SeqVec | {vector_sizes: [4,4,4,4,1,1]} | T | F | N/A | N/A |
| S5 | ParVec | {vector_sizes: [4,4,4,4,1,1]} | T | F | N/A | N/A |
| S6 | Im2colLowering | {} | F | F | N/A | N/A |
| S7 | TilingPar | {tile_sizes: [16,16,4,4,0,0]} | T | T | (pending) | - |
| S8 | ThreadPar | {num_threads: 28} | T | T | (pending) | - |
| S9 | Packing | {packed_sizes: [16,16,0,0,0,0]} | T | T | (pending) | - |
| S10 | Unrolling | {unroll_factor: 4} | T | T | (pending) | - |

**Applicable:** Tiling, LoopInterchange, Promotion, TilingPar, ThreadPar, Packing, Unrolling (7/10)
**Not applicable:** SeqVec (post=F), ParVec (post=F), Im2col (pre=F)

### Add (112x112x120x150, base=338.45ms, torch=68.09ms)

| # | Action | Parameters | Pre | Post | Time (ms) | Speedup |
|---|--------|-----------|-----|------|-----------|---------|
| S1 | Tiling | {tile_sizes: [8,8,8,0]} | T | T | (pending) | - |
| S2 | LoopInterchange | {permutation: [1,0,2,3]} | T | T | (pending) | - |
| S3 | Promotion | {operands: [0,1,2]} | T | T | (pending) | - |
| S4 | SeqVec | {vector_sizes: [4,4,4,2]} | T | T | (pending) | - |
| S5 | ParVec | {vector_sizes: [4,4,4,2]} | T | T | (pending) | - |
| S6 | Im2colLowering | {} | F | F | N/A | N/A |
| S7 | TilingPar | {tile_sizes: [8,8,8,0]} | T | T | (pending) | - |
| S8 | ThreadPar | {num_threads: 28} | T | T | (pending) | - |
| S9 | Packing | {packed_sizes: [8,8,8,0]} | T | T | (pending) | - |
| S10 | Unrolling | {unroll_factor: 4} | T | T | (pending) | - |

**Applicable:** Tiling, LoopInterchange, Promotion, SeqVec, ParVec, TilingPar, ThreadPar, Packing, Unrolling (9/10)
**FINDING: LoopInterchange WORKS on add** — previous registry incorrectly blocked it!

### Relu (128x128x56x56, base=65.37ms, torch=12.22ms)

| # | Action | Parameters | Pre | Post | Time (ms) | Speedup |
|---|--------|-----------|-----|------|-----------|---------|
| S1 | Tiling | {tile_sizes: [16,16,8,0]} | T | T | (pending) | - |
| S2 | LoopInterchange | {permutation: [1,0,2,3]} | T | T | (pending) | - |
| S3 | Promotion | {operands: [0,1,2]} | T | T | (pending) | - |
| S4 | SeqVec | {vector_sizes: [4,4,4,4]} | T | T | (pending) | - |
| S5 | ParVec | {vector_sizes: [4,4,4,4]} | T | T | (pending) | - |
| S6 | Im2colLowering | {} | F | F | N/A | N/A |
| S7 | TilingPar | {tile_sizes: [16,16,8,0]} | T | T | 12.20 | 5.36x |
| S8 | ThreadPar | {num_threads: 28} | T | T | 10.12 | 6.46x |
| S9 | Packing | {packed_sizes: [16,16,8,0]} | T | T | (pending) | - |
| S10 | Unrolling | {unroll_factor: 4} | T | T | (pending) | - |

**Applicable:** Tiling, LoopInterchange, Promotion, SeqVec, ParVec, TilingPar, ThreadPar, Packing, Unrolling (9/10)
**FINDING: LoopInterchange WORKS on relu** — matches add pattern (4D elementwise parallel, linalg.generic)
**FINDING: ThreadPar beats PyTorch** — 10.12ms vs 12.22ms (1.21x vs torch)

## Phase 1: Applicability Summary

| Action | Matmul | Conv2d | Pooling | Add | Relu |
|--------|--------|--------|---------|-----|------|
| Tiling | ✓ | ✓ | ✓ | ✓ | ✓ |
| LoopInterchange | ✓ | ✓ | ✓ | ✓ | ✓ |
| Promotion | ✓ | ✓ | ✓ | ✓ | ✓ |
| SeqVec | ✓ | ✗(pre) | ✗(post) | ✓ | ✓ |
| ParVec | ✓ | ✗(pre) | ✗(post) | ✓ | ✓ |
| Im2col | ✗(pre) | ✓ | ✗(pre) | ✗(pre) | ✗(pre) |
| TilingPar | ✓ | ✓ | ✓ | ✓ | ✓ |
| ThreadPar | ✓ | ✓ | ✓ | ✓ | ✓ |
| Packing | ✓ | ✓ | ✓ | ✓ | ✓ |
| Unrolling | ✓ | ✓ | ✓ | ✓ | ✓ |

**Key correction from v53:** LoopInterchange works on add and relu (v53 incorrectly blocked it).

## Phase 2: Pairwise Compositions (Key Results)

### Matmul Compositions (agent-validated on 3 kernels: 128x128x128, 512x256x1536, 1024x1024x256)

| Schedule Path | Time (ms) | Speedup vs Base | Notes |
|---------------|-----------|-----------------|-------|
| TilingPar[32,32,0] → SeqVec[8,8,8] | 1.78 | 275.9x | Best 2-step (medium) |
| TilingPar[64,64,0] → Tiling[16,16,16] → SeqVec[4,4,4] | 1.81 | 271.1x | Best 3-step (medium) |
| TilingPar[32,32,0] → Tiling[8,8,8] → SeqVec[8,8,8] | 2.12 | 231.4x | 3-step alt params |
| TilingPar[64,64,0] → SeqVec[4,4,4] | 2.76 | 177.8x | |
| TilingPar[64,64,0] → ParVec[4,4,4] | 2.94 | 166.9x | SeqVec beats ParVec |
| LI[1,0,2] → TilingPar[64,64,0] → ParVec[4,4,4] | 3.35 | 146.5x | LI adds overhead |
| TilingPar[64,64,0] → Tiling[16,16,16] | 3.79 | 129.5x | Non-terminal |
| ThreadPar[8] → Tiling[16,16,16] → SeqVec[4,4,4] | 4.00 | 122.7x | Alt parallel strat |
| ParVec[4,4,4] | 8.47 | 56.4x | Best single action |
| Packing[32,32,32] → TilingPar[32,32,0] | 11.47 | 41.6x | v55 NEW path |
| Packing → ParVec | N/A | N/A | Pre=F after Packing |

**Cross-kernel validation (3-step: PTile→Tiling→SeqVec):**
- Small (128x128x128): 0.121ms (22.1x, 0.73x vs torch)
- Medium (512x256x1536): 1.81ms (271.1x, 0.31x vs torch)
- Large (1024x1024x256): 3.45ms (195.3x, 0.23x vs torch)

### Conv2d Compositions (agent-validated on 3 kernels: 128x128x7x7, 128x128x14x14, 256x192x15x15)

| Schedule Path | Time (ms) | Speedup vs Base | Notes |
|---------------|-----------|-----------------|-------|
| **ThreadPar(28) → Tiling** | **5.34** | **32.6x** | Best (medium), agent-validated |
| TilingPar → Tiling | 5.58 | 31.2x | Runner-up |
| ThreadPar(28) | 7.04 | 24.7x | Best single action |
| TilingPar | 7.46 | 23.3x | |
| TilingPar → Im2col | 8.37 | 20.8x | |
| Im2col → TilingPar | 10.99 | 15.8x | |
| Im2col → ThreadPar | 10.56 | 16.5x | |

**Cross-kernel validation (ThreadPar→Tiling):**
- Small (128x128x7x7→48x3x3): 2.91ms (27.5x, 0.48x vs torch)
- Medium (128x128x14x14→192x7x7): 5.34ms (32.6x, 0.16x vs torch)
- Large (256x192x15x15→32x8x8): 3.46ms (37.7x, 0.51x vs torch)

**Conv2d composability (8x8 matrix, key findings):**
- Tiling: 7/7 as first (most composable — preserves conv_2d_nchw_fchw within scf.for)
- Packing: 0/7 as first (terminal for conv2d — transforms IR structure)
- Unrolling: 0/7 as first (terminal for conv2d)
- Im2col after LI: FAILS (Im2col requires conv_2d_nchw_fchw, not generic)

### Pooling Compositions (agent-validated on 3 kernels: 128x128x7x7, 128x128x56x56, 128x128x112x112)

| Schedule Path | Time (ms) | Speedup vs Base | vs PyTorch | Notes |
|---------------|-----------|-----------------|------------|-------|
| **ThreadPar(28)** | **2.34** | **10.16x** | **4.67x** | Best (medium) |
| TilingPar | 3.58 | 6.64x | 3.05x | |
| LI → ThreadPar | 2.87 | 8.29x | 3.81x | LI adds overhead |
| ThP → Tiling | 3.34 | 7.12x | 3.27x | Composition worse |
| ThP → TilingPar | 3.35 | 7.10x | 3.26x | Composition worse |
| TP → ThreadPar | 3.47 | 6.85x | 3.15x | Two-level worse |
| LI → TilingPar | 3.98 | 5.97x | 2.75x | |
| Tiling alone | 51.85 | 0.46x | 0.21x | **HARMFUL** |
| Promotion alone | 141.51 | 0.17x | 0.08x | **HARMFUL** |
| Packing alone | 402.66 | 0.06x | 0.03x | **HARMFUL** |

**Cross-kernel validation (ThreadPar):**
- Small (128x128x7x7, k=3x3): 0.175ms (9.77x, **1.32x vs torch**)
- Medium (128x128x56x56, k=1x1): 2.34ms (10.16x, **4.67x vs torch**)
- Large (128x128x112x112, k=1x1): 10.03ms (8.75x, **4.02x vs torch**)

**Key finding:** Single ThreadPar is universally optimal for pooling. All compositions
degrade performance vs ThreadPar alone. Non-parallel transforms (Tiling, Promotion,
Packing) are actively harmful (0.06-0.46x).

### Add Compositions (agent-validated on 3 kernels: 112x14x112x15, 112x112x120x150, 28x56x14x240)

| Schedule Path | Time (ms) | Speedup vs Base | vs PyTorch | Notes |
|---------------|-----------|-----------------|------------|-------|
| ThreadPar[28] | 52.20 | 6.48x | **1.30x** | Best single (medium) |
| TilingPar[8,0,8,0] | 52.92 | 6.39x | **1.29x** | Tied for best (medium) |
| TilingPar → ThreadPar | 57.04 | 5.93x | **1.19x** | Marginal 2-step overhead |
| TilingPar → SeqVec | 74.33 | 4.55x | 0.92x | |
| Tiling → ThreadPar | 104.89 | 3.23x | 0.65x | |
| ThreadPar → SeqVec | 130.17 | 2.60x | 0.52x | |
| LI → ThreadPar (full reverse) | 338.44 | 1.00x | 0.20x | Full dim reversal harmful |
| SeqVec alone | 2905.29 | 0.12x | 0.02x | **Vectorization harmful!** |

**Cross-kernel validation:**
- Small (112x14x112x15): ThreadPar=0.77ms (4.43x, 0.66x vs torch)
- Medium (112x112x120x150): ThreadPar=52.20ms (6.48x, 1.30x vs torch)
- Large (28x56x14x240): TilingPar=0.84ms (7.37x, **2.36x vs torch**)

### Relu Compositions (agent-validated on 3 kernels: 128x128x7x7, 128x128x56x56, 256x256x56x56)

| Schedule Path | Time (ms) | Speedup vs Base | vs PyTorch | Notes |
|---------------|-----------|-----------------|------------|-------|
| ThreadPar[28] | 10.19 | 6.45x | **1.20x** | Best (medium), beats PyTorch |
| ThreadPar → SeqVec[1,4,8,8] | 11.37 | 5.78x | 1.07x | Vec adds overhead |
| TilingPar[16,16,8,8] | 12.16 | 5.40x | 1.00x | |
| TilingPar → SeqVec[4,4,4,4] | 16.23 | 4.05x | 0.75x | Vec degrades perf |
| ParVec[4,4,4,4] alone | 91.76 | 0.72x | 0.13x | **HARMFUL** |
| SeqVec[4,4,4,4] alone | 181.27 | 0.36x | 0.07x | **HARMFUL** |
| Packing[16,16,8,8] alone | 680.91 | 0.10x | 0.02x | **HARMFUL** |

**Cross-kernel validation (ThreadPar):**
- Small (128x128x7x7): 0.35ms (1.79x, **1.91x vs torch**)
- Medium (128x128x56x56): 10.19ms (6.45x, **1.20x vs torch**)
- Large (256x256x56x56): 42.49ms (6.20x, **1.10x vs torch**)

**Key finding:** Vectorization is HARMFUL for relu. SeqVec (0.36x) and ParVec (0.72x)
both cause slowdowns. Even after parallelization, adding vectorization degrades performance
(ThreadPar 10.19ms → ThreadPar→SeqVec 11.37ms). The LLVM backend already auto-vectorizes
the inner loops, making explicit MLIR vectorization redundant overhead.

## Phase 3: Multi-Step Schedules

### Composability Matrix (All Families)

All families share a clean **7+2 partition**: 7 structural actions (Tiling, LoopInterchange,
Promotion, ParallelizationTiling, ParallelizationThreads, Packing, Unrolling) compose freely
with each other. 2 vectorization actions (VectorizationSeq, VectorizationPar) are **TERMINAL**:
they accept any predecessor but reject all successors (tag moves from linalg op to scf loop).

### Key 3-Step Findings

**Matmul:** The 3-level hierarchy `PTile → Tiling → SeqVec` is the universal best:
- Medium (512x256x1536): 1.81ms (271x) — competitive with 2-step PTile→SeqVec (1.78ms, 276x)
- Small (128x128x128): 0.121ms (22x) — beats 2-step (0.148ms, 18x) by 18%
- Large (1024x1024x256): 3.45ms (195x) — beats 2-step (5.30ms, 127x) by 54%
- Pattern: parallel decomposition → cache-level blocking → register vectorization (mirrors BLAS)
- SeqVec outperforms ParVec for matmul: SeqVec produces scf.for loops with better codegen

**Conv2d:** ThreadPar(28)→Tiling is the best at ~32x avg (27.5-37.7x across kernels),
beating PTile→Tiling (~27x avg). ThreadPar maps batch dimension directly to 28 cores,
then Tiling blocks F and C dims for cache locality. Packing and Unrolling are terminal
for conv2d (0/7 composability as first action).

**Pooling:** Single ThreadPar(28) at 10.16x (4.67x vs torch) is universally optimal.
All compositions degrade performance vs ThreadPar alone. Non-parallel transforms
(Tiling 0.46x, Promotion 0.17x, Packing 0.06x) are actively harmful.

**Add:** Simple parallelization is best. ThreadPar alone (6.48x, 1.30x vs torch) or
TilingPar alone (6.39x, 1.29x vs torch) are the top schedules. Multi-step schedules
add overhead that exceeds their benefit for elementwise operations.
**Critical:** Vectorization is actively harmful for add — SeqVec alone makes performance
8.6x WORSE than unoptimized base (2905ms vs 338ms).

**Relu:** Matches add pattern. ThreadPar (6.46x, 1.21x vs torch) beats PyTorch.

### Vectorization Terminality

VectorizationSeq and VectorizationPar are confirmed TERMINAL across all families:
after vectorization, the tag moves from the linalg op to an scf loop, and subsequent
actions cannot find the target. Vectorization must always be the LAST action in a schedule.

## Phase 4: Best Schedule Shapes per Family

### Final Rankings (Agent-Validated)

| Rank | Matmul | Conv2d | Pooling | Add | Relu |
|------|--------|--------|---------|-----|------|
| 1 | PTile→SeqVec | **ThreadPar→Tile** | **ThreadPar** | ThreadPar | ThreadPar |
| 2 | PTile→Tile→SeqVec | PTile→Tile | PTile | PTile | PTile |
| 3 | PTile→VecPar | LI→PTile→Tile | *(no 3rd)* | PTile→ThreadPar | *(no 3rd)* |
| 4 | VecPar | LI→PTile | | VecPar | |
| 5 | LI→VecPar | ThreadPar | | PTile→VecSeq | |

### v55 New Findings

1. **SeqVec outperforms ParVec for matmul** (276x vs 193x with optimal params). SeqVec
   produces scf.for sequential loops that enable better backend codegen than scf.forall.

2. **3-level PTile→Tiling→SeqVec is the universal best for matmul**. Wins on small
   and large kernels; competitive on medium. Mirrors classical BLAS optimization.

3. **ThreadPar→Tiling is the best conv2d schedule** (~32x avg, agent-validated on 3 kernels).
   ThreadPar maps batch dim to cores, then Tiling blocks filter/channel dims for cache.
   Packing and Unrolling are terminal for conv2d (0/7 composability as first action).

4. **Packing → ParallelizationTiling (matmul):** 11.47ms (41.6x). Packing restructures
   data layout for better cache behavior before parallelization. New v55 schedule path.

5. **Parallelization-only is optimal for elementwise ops (add/relu)**. Both ThreadPar
   and TilingPar beat PyTorch (1.3x, 2.36x on large add). Vectorization and tiling
   add overhead that exceeds their benefit for simple elementwise operations.

6. **LoopInterchange works on add/relu** — v53 incorrectly blocked it. However, agent
   testing shows LI with full dimension reversal can be harmful (338ms on add). The
   benefit is permutation-dependent; partial swaps [1,0,2,3] can help but full
   reversal [3,2,1,0] destroys memory locality.

7. **Unrolling shows negligible benefit as a single action** across all families (~1.0x).
   Not included in any top schedule path.

8. **Packing after vectorization fails (pre=F):** Packing transforms the op structure
   (matmul → 6D generic), making subsequent vectorization preconditions fail.

## Output Artifacts

- **ACTION_DEPENDENCIES:** Written to `llm_action/src/actions/v55/registry.py`
- **SCHEDULE_GRAPH:** Written to `llm_action/src/actions/v55/registry.py`
- Both structures empirically validated via parallel agent exploration on v55 MCP tools
- Cross-kernel validated on 3 kernel sizes per family (small/medium/large)
- Date: 2026-08-15
