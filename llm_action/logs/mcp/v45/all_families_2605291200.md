# MLIR Schedule Exploration Log: All Kernel Families
- Action Version: v45
- Kernels: matmul_512_512_1024, conv_2d_nchw_fchw_128_64_7_7_64_1_1_7_7, pooling_nchw_max_128_128_14_14_7_4_4, add_240_14_28_224, relu_128_256_14_14
- Hardware: Intel Xeon E5-2680 v4 (Broadwell), AVX2, 28 cores
- Date: 2026-05-29

## Baselines

| Kernel | MLIR Base (ms) | PyTorch (ms) |
|--------|---------------|-------------|
| matmul_512_512_1024 (M=512, K=512, N=1024) | 654.10 | 0.74 |
| conv_2d_nchw_fchw_128_64_7_7_64_1_1_7_7 (N=128,C=64,H=7,W=7,F=64,KH=1,KW=1,OH=7,OW=7) | 24.52 | 0.20 |
| pooling_nchw_max_128_128_14_14_7_4_4 (N=128,C=128,H=14,W=14,KH=7,KW=7,OH=4,OW=4) | 24.54 | 0.81 |
| add_240_14_28_224 (A=240,B=14,C=28,D=224) | 31.21 | 7.24 |
| relu_128_256_14_14 (N=128,C=256,H=14,W=14) | 7.96 | 1.91 |

## Phase 1: Single Actions — Applicability Matrix

### Pre/Post results per action per family

| Action | matmul Pre/Post | conv2d Pre/Post | pooling Pre/Post | add Pre/Post | relu Pre/Post |
|--------|----------------|-----------------|------------------|-------------|--------------|
| Tiling | T/T | T/T | T/T | T/T | T/T |
| LoopInterchange | T/T | T/T | T/T | T/T | T/T |
| Packing | T/T | T/T | **F/F** | T/T | T/T |
| Promotion | T/T | T/T | T/T | T/T | T/T |
| VectorizationSeq | T/T | T/**F** | T/**F** | T/T | T/T |
| VectorizationPar | T/T | T/T | T/T | T/T | T/T |
| LoopUnrolling | T/T | T/T | T/T | T/T | T/T |
| ParallelizationTile | T/T | T/T | T/T | T/T | T/T |
| ParallelizationThreads | T/T | T/T | T/T | T/T | T/T |
| Im2colLowering | **F/F** | T/T | **F/F** | **F/F** | **F/F** |

### Probe Parameters Used

**matmul_512_512_1024** (dims: M=512, K=512, N=1024, 3 loops):
- Tiling: [64,64,64], Interchange: [1,0,2], Packing: [32,32,32], Promotion: [64,64,64]
- VecSeq: [8,8,8], VecPar: [8,8,8], Unroll: 4, ParTile: [16,16,0], ParThreads: 28

**conv_2d_nchw_fchw_128_64_7_7_64_1_1_7_7** (dims: N=128,F=64,OH=7,OW=7,C=64,KH=1,KW=1, 7 loops):
- Tiling: [16,16,7,7,16,0,0], Interchange: [1,0,2,3,4,5,6], Packing: [16,16,0,0,16,0,0]
- Promotion: [16,16,7,7,16,0,0], VecSeq: [1,4,1,1,4,1,1], VecPar: [1,4,1,1,4,1,1]
- Unroll: 4, ParTile: [16,16,7,7,0,0,0], ParThreads: 28

**pooling_nchw_max_128_128_14_14_7_4_4** (dims: N=128,C=128,OH=4,OW=4,KH=7,KW=7, 6 loops):
- Tiling: [16,16,2,2,0,0], Interchange: [1,0,2,3,4,5], Packing: [16,16,2,2,0,0]
- Promotion: [16,16,2,2,0,0], VecSeq: [1,4,2,2,7,7], VecPar: [1,4,2,2,1,1]
- Unroll: 4, ParTile: [16,16,2,2,0,0], ParThreads: 28

**add_240_14_28_224** (dims: 240×14×28×224, 4 parallel loops):
- Tiling: [16,14,4,16], Interchange: [1,0,2,3], Packing: [16,0,4,16]
- Promotion: [16,14,4,16], VecSeq: [4,2,4,4], VecPar: [4,2,4,4]
- Unroll: 4, ParTile: [16,14,4,16], ParThreads: 28

**relu_128_256_14_14** (dims: 128×256×14×14, 4 parallel loops):
- Tiling: [16,16,14,14], Interchange: [1,0,2,3], Packing: [16,16,0,0]
- Promotion: [16,16,14,14], VecSeq: [4,4,2,2], VecPar: [4,4,2,2]
- Unroll: 4, ParTile: [16,16,14,14], ParThreads: 28

### Execution Times for matmul Single Actions

| # | Action | Parameters | Pre | Post | Time (ms) | vs Base |
|---|--------|-----------|-----|------|-----------|---------|
| S1 | Tiling | [64,64,64] | T | T | 576.02 | 1.14x |
| S2 | LoopInterchange | [1,0,2] | T | T | 734.74 | 0.89x |
| S3 | Packing | [32,32,32] | T | T | 135.72 | 4.82x |
| S4 | Promotion | [64,64,64] | T | T | 286.42 | 2.28x |
| S5 | VecSeq | [8,8,8] | T | T | (pending) | - |
| S6 | VecPar | [8,8,8] | T | T | (pending) | - |
| S7 | LoopUnrolling | 4 | T | T | (pending) | - |
| S8 | ParTile | [16,16,0] | T | T | (pending) | - |
| S9 | ParThreads | 28 | T | T | (pending) | - |
| S10 | Im2col | {} | F | F | N/A (not applicable) | - |

### Key Observations from Phase 1

1. **VecSeq fails postcondition on conv2d and pooling** — named ops (conv_2d_nchw_fchw, pooling_nchw_max) can't be directly vectorized sequentially. VecPar works on all.
2. **Packing fails precondition on pooling** — pooling ops with strides > 1 don't support packing.
3. **Im2col only applies to conv_2d_nchw_fchw** — precondition rejects all other op types.
4. **Promotion converts to memref** — all other actions remain in tensor semantics.
5. **Packing provides best single-action speedup on matmul** (4.82x with [32,32,32]).

## Phase 2: Pairwise Compositions (matmul_512_512_1024)

### Full Composability Matrix (Pre/Post)

| After \ Then | Tile | Interchange | Pack | Promo | VecSeq | VecPar | Unroll | ParTile | ParThreads |
|---|---|---|---|---|---|---|---|---|---|
| **Tiling** | T/T | T/T | T/T | T/T | T/T | T/T | T/T | T/T | T/T |
| **Interchange** | T/T | T/T | T/T | T/T | T/T | T/T | T/T | T/T | T/T |
| **Packing** | T/T | - | - | T/T | **F/F** | - | - | - | - |
| **Promotion** | T/T | T/T | T/T | T/T | T/T | T/T | T/T | T/T | T/T |
| **VecSeq** | T/**F** | T/**F** | T/**F** | **F/F** | **F/F** | **F/F** | T/**F** | **F/F** | **F/F** |
| **VecPar** | T/**F** | - | - | T/**F** | T/**F** | - | - | - | - |
| **Unrolling** | - | - | - | - | - | - | - | - | - |
| **ParTile** | T/T | - | - | T/T | T/T | - | - | - | - |
| **ParThreads** | - | - | - | - | - | - | - | - | - |

*T/T = Pre passes, Post passes. T/F = Pre passes but Post fails (no-op). F/F = Pre fails. "-" = untested (inferred from structural similarity).*

### Key Findings from Phase 2

1. **FULLY COMPOSABLE first actions** (all 9 follow-ups work):
   - Tiling (9/9 tested)
   - LoopInterchange (9/9 tested)
   - Promotion (9/9 tested)
   - ParallelizationTiling (3/3 probes: Tiling T/T, VecSeq T/T, Promotion T/T)

2. **TERMINAL actions** (no follow-ups work):
   - **VectorizationSequential**: 0/9 — consumes linalg op, replaces with vector ops. Tag moves to scf.for. All follow-ups either fail precondition (Promotion, VecSeq, VecPar, ParTile, ParThreads) or pass precondition but produce no change/fail postcondition (Tiling, Interchange, Packing, Unrolling).
   - **VectorizationParallel**: 0/3 probes — preserves inner linalg.matmul inside forall, but all follow-ups pass precondition yet fail postcondition (Tiling T/F, VecSeq T/F, Promotion T/F). Effectively terminal.

3. **PARTIALLY COMPOSABLE**:
   - **Packing**: Tiling T/T, Promotion T/T, but **VecSeq F/F** (precondition fails on packed 6D generic). Packing blocks VecSeq.

4. **Structural inference for untested rows**:
   - LoopUnrolling creates multiple copies of linalg ops in scf.for → structurally similar to Tiling (likely composable)
   - ParallelizationThreads creates forall with linalg inside → structurally identical to ParallelizationTiling (likely composable)

### Composability Rules

- **VecSeq and VecPar must be LAST** in any schedule (terminal actions)
- **Packing → VecSeq is blocked** (precondition fails on 6D generic)
- **VecSeq is not applicable** to conv2d and pooling named ops (postcondition fails from Phase 1)
- **Packing is not applicable** to pooling (precondition fails from Phase 1)
- **Im2col is only applicable** to conv_2d_nchw_fchw
- **Promotion converts to memref** but ALL follow-up actions still work (inner linalg op preserved)
- **Im2col → VecSeq WORKS** on the resulting generic (unlike direct VecSeq on named conv_2d_nchw_fchw)
- **Im2col → Packing FAILS** (Pre=F on the 4D generic produced by im2col)

### Verified Remaining Rows (Session 2)

| After \ Then | Tile | VecSeq | Promo | Result |
|---|---|---|---|---|
| **LoopUnrolling** | T/T | T/T | T/T | FULLY COMPOSABLE (confirmed) |
| **ParThreads** | T/T | T/T | T/T | FULLY COMPOSABLE (confirmed) |

### Conv2d Im2col Compositions

| After Im2col → | Tiling | Packing | Promotion | VecSeq | VecPar |
|---|---|---|---|---|---|
| Pre/Post | T/T | **F/F** | T/T | T/T | T/T |

Im2col converts conv_2d_nchw_fchw into a 4D linalg.generic (3 parallel + 1 reduction, batch matmul pattern).
Key finding: VecSeq **succeeds** on the im2col-lowered generic even though it **fails** on the original named op.
Packing fails on the resulting generic — the non-standard indexing maps block the packing precondition.

## Phase 3-4: Multi-Step Schedule Optimization (Parallel Agents)

### Methodology
5 parallel agents (one per family) each explored 3 representative kernels
(small/medium/large) with deep multi-step schedules (depth 3-5), measuring
execution time via execute_mlir_code and comparing against PyTorch baselines.

### Results Summary

| Family | Best Schedule | vs PyTorch | vs MLIR Base | Beats PyTorch? |
|--------|--------------|------------|--------------|----------------|
| matmul | ParTile→Tile→VecSeq | 1.34x (128³) | 321x (512×512×1024) | YES (small) |
| conv2d | ParTile→ParThreads→Tile→Promo | 0.34x best | 42x | No |
| pooling | ParTile→Promotion | 2.79x (large) | 37x | YES (all 3!) |
| add | ParThreads→VecSeq | 1.80x (medium) | 7.9x | YES (med/large) |
| relu | ParTile→Tile→VecSeq | 1.25x (medium) | 5.4x | YES (medium) |

### Key Findings

**matmul (187 train kernels):**
- Champion: ParTile[32,32,0]→Tile[32,32,32]→VecSeq[8,8,8] = 2.09ms (321x over 670ms base)
- BEATS PyTorch on 128×128×128: 0.116ms vs 0.156ms (1.34x faster)
- VecSeq[8,8,8] is the sweet spot for f64 AVX2 (256-bit = 4×f64)
- ParallelizationTiling is essential — adds 10-30x additional speedup

**conv_2d_nchw_fchw (278 train kernels):**
- Champion: ParTile[8,8,0,0,0,0,0]→ParThreads[28]→Promo[0,0,0,0,8,0,0] = 0.581ms (42x)
- Named-op paths dominate; Im2col paths are consistently SLOWER
- VecSeq on Im2col output fails due to dynamic shapes (49 not power of 2)
- ParTile→ParThreads is the dominant parallelization strategy for conv

**pooling_nchw_max (250 train kernels):**
- Champion: ParTile[4,4,4,4]→Promo[4,4,4,4] = 0.655ms (37x, BEATS PyTorch 1.23x)
- BEATS PyTorch on ALL 3 kernels tested (1.23x, 2.79x, 1.50x)
- ParThreads[28] alone: 19.41ms on large spatial (BEATS PyTorch 2.79x)
- Small tile sizes (4,4) work best for batch/channel dims

**add (271 train kernels):**
- Champion: ParThreads[28]→VecSeq[1,1,1,32] = 3.93ms (7.9x, BEATS PyTorch 1.80x)
- Memory-bound: parallelization is the key, vectorization adds modest gain
- VecSeq parameters: use scalar leading dims, maximize innermost vector width
- Tiling HURTS performance — extra loop overhead outweighs cache benefit

**relu (149 train kernels):**
- Champion: ParTile[16,16,0,0]→Tile[4,4,0,0]→VecSeq[1,1,14,14] = 1.47ms (5.4x, BEATS PyTorch 1.25x)
- ParTile→VecPar best for large kernels (54ms, 4.77x over base)
- Small kernels (<2M elements) dominated by overhead, PyTorch wins

## Synthesized Structures

### ACTION_DEPENDENCIES (cross-family denylist)

Terminal actions block all successors:
- VectorizationSequential → [all 10 actions]
- VectorizationParallel → [all 10 actions]
- Packing → [VectorizationSequential] (Pre=F on packed code)

### SCHEDULE_GRAPH (per-family allowlist, optimized)

Written to `llm_action/src/actions/v45/registry.py`.

**matmul** (11 paths): ParTile→Tile→VecSeq (champion), ParTile→VecSeq, ParTile→Tile→Promo, ParTile→Tile→Promo→VecSeq, ParThreads→Tile→VecSeq, ParThreads→Tile→Promo→VecSeq, Packing, Tile→VecSeq, Tile→Pack, Tile→Promo, Interchange→Tile→VecSeq.

**conv_2d_nchw_fchw** (11 paths): ParTile→ParThreads→Tile→Promo (champion), ParTile→ParThreads→Promo, ParTile→ParThreads→Tile, ParTile→ParThreads, ParTile→Tile→VecPar, ParTile→Tile→Promo→VecPar, Im2col→ParTile→ParThreads, Im2col→Tile→VecSeq, Im2col→Tile→Promo, Tile→VecPar, Tile→Promo.

**pooling_nchw** (9 paths): ParTile→Promo (champion, beats PyTorch), ParThreads, ParTile→Tile, ParTile→Tile→Promo, ParTile→VecPar, ParTile→ParThreads→Tile→Promo, ParThreads→Tile→VecPar, Tile→VecPar, Tile→Promo.

**add** (8 paths): ParThreads→VecSeq (champion, beats PyTorch), ParThreads→Tile→VecSeq, ParTile→VecSeq, ParTile→VecPar, ParTile→Tile→VecSeq, ParTile→Tile→VecPar, ParThreads→VecPar, Tile→VecSeq.

**relu** (8 paths): ParTile→Tile→VecSeq (champion, beats PyTorch), ParTile→VecSeq, ParTile→VecPar, ParThreads→VecSeq, ParTile→Tile→VecPar, ParTile→ParThreads→VecSeq, ParTile→ParThreads→VecPar, Tile→VecSeq.

**default** (7 paths): ParTile→Tile→VecSeq, ParTile→VecSeq, ParThreads→VecSeq, ParTile→Promo, Tile→VecSeq, Tile→Promo, Packing.
