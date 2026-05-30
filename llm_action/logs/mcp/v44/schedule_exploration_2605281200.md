# MLIR Schedule Exploration Log: All Kernel Families (dataset_ml)
- Action Version: v44
- Families: matmul, conv2d, pooling, add, relu
- Hardware: Intel Xeon E5-2680 v4 (Broadwell), AVX2, 28 cores
- Date: 2026-05-28

## Phase 0: Baselines

### Representative Kernels (Phases 1-2: 1 per family)

| Family | Kernel | MLIR Base (ms) | PyTorch (ms) |
|--------|--------|---------------|-------------|
| matmul | matmul_128_128_128 (128x128 @ 128x128) | 2.645 | 0.088 |
| conv2d | conv_2d_nchw_fchw_128_32_7_7_32_3_3_3_3 | 13.093 | 0.310 |
| pooling | pooling_nchw_max_128_128_14_14_7_4_4 | 24.551 | 0.894 |
| add | add_112_112_14_15 | 3.379 | 0.551 |
| relu | relu_256_32_28_28 | 8.004 | 2.030 |

## Phase 1: Single Actions

### matmul_128_128_128 (dims: M=128, N=128, K=128)

| Action | Parameters | Pre | Post | Time (ms) | Speedup |
|--------|-----------|-----|------|-----------|---------|
| Tiling | [4,4,4] | T | T | 1.718 | 1.54x |
| LoopInterchange | [1,0,2] | T | T | 2.761 | 0.96x |
| Packing | [4,4,4] | T | T | 1.722 | 1.54x |
| Promotion | [4,4,4] | T | T | 1.022 | 2.59x |
| VecSeq | [4,4,4] | T | T | 0.394 | 6.71x |
| VecPar | [4,4,4] | T | T | 2.518 | 1.05x |
| LoopUnrolling | 4 | T | T | 2.671 | 0.99x |
| ParallelTiling | [4,4,0] | T | T | **0.190** | **13.92x** |
| ParallelThreads | 4 | T | T | 0.749 | 3.53x |
| Im2col | {} | **F** | - | - | N/A (not conv2d) |

### conv_2d_nchw_fchw_128_32_7_7_32_3_3_3_3 (dims: N=128,C=32,H=7,W=7,F=32,KH=3,KW=3,OH=3,OW=3)

| Action | Parameters | Pre | Post | Time (ms) | Speedup |
|--------|-----------|-----|------|-----------|---------|
| Tiling | [4,4,0,0,4,0,0] | T | T | 9.049 | 1.45x |
| LoopInterchange | [1,0,2,3,4,5,6] | T | T | - | - |
| Packing | [4,4,0,0,4,0,0] | T | T | - | - |
| Promotion | [4,4,0,0,4,0,0] | T | T | 10.461 | 1.25x |
| VecSeq | [4,4,1,1,4,1,1] | T | **F** | - | FAIL |
| VecPar | [4,4,1,1,4,1,1] | T | T | - | - |
| LoopUnrolling | 4 | T | T | - | - |
| ParallelTiling | [4,4,0,0,0,0,0] | T | T | **0.574** | **22.8x** |
| ParallelThreads | 4 | T | T | - | - |
| Im2col | {} | T | T | 39.275 | 0.33x (slower!) |

### pooling_nchw_max_128_128_14_14_7_4_4 (dims: N=128,C=128,H=14,W=14,KH=7,KW=7,OH=4,OW=4)

| Action | Parameters | Pre | Post | Time (ms) | Speedup |
|--------|-----------|-----|------|-----------|---------|
| Tiling | [4,4,4,4,0,0] | T | T | - | - |
| VecSeq | [4,4,4,4,1,1] | T | **F** | - | FAIL |
| VecPar | [4,4,4,4,1,1] | T | T | - | - |
| ParallelTiling | [4,4,4,4,0,0] | T | T | **0.978** | **25.1x** |

### add_112_112_14_15 (dims: 112x112x14x15, all parallel)

| Action | Parameters | Pre | Post | Time (ms) | Speedup |
|--------|-----------|-----|------|-----------|---------|
| Tiling | [4,4,0,0] | T | T | - | - |
| VecSeq | [4,4,2,1] | T | T | 7.642 | 0.44x (slower!) |
| VecPar | [4,4,2,1] | T | T | - | - |
| ParallelTiling | [4,4,0,0] | T | T | **0.639** | **5.29x** |

### relu_256_32_28_28 (dims: 256x32x28x28, all parallel)

| Action | Parameters | Pre | Post | Time (ms) | Speedup |
|--------|-----------|-----|------|-----------|---------|
| Tiling | [4,4,4,4] | T | T | - | - |
| VecSeq | [4,4,4,4] | T | T | 14.956 | 0.54x (slower!) |
| VecPar | [4,4,4,4] | T | T | - | - |
| ParallelTiling | [4,4,4,4] | T | T | **1.239** | **6.46x** |

### Phase 1 Key Findings
1. **ParallelTiling is the strongest single action across ALL families** (6-25x speedup)
2. **VecSeq fails Post on conv2d and pooling** (structured ops with windowed access)
3. **VecSeq standalone is SLOWER on add/relu** (overhead without parallelization)
4. **Im2col is slower on small conv** (reshape overhead dominates)
5. **Im2col only applies to conv2d** (Pre=F for others)

## Phase 2: Pairwise Composability (matmul_128_128_128)

### Composability Matrix

Legend: T=Post passes, F=Post fails, x=Pre fails, -=not tested (inferred from terminal row)

| First ↓ \ Then → | Tile | LI | Pack | Promo | VecSeq | VecPar | LU | ParTile | ParThrd |
|---|---|---|---|---|---|---|---|---|---|
| **Tiling** | T | T | T | T | T | T | T | T | T |
| **LoopInterchange** | T | T | T | T | T | T | T | T | T |
| **Packing** | T | - | - | - | - | - | - | - | - |
| **Promotion** | T | T | T | **x** | T | T | T | T | T |
| **VecSeq** | **F** | - | - | - | - | - | - | **F** | - |
| **VecPar** | **F** | - | - | - | - | - | - | **F** | - |
| **LoopUnrolling** | T | T | **F** | T | T | T | - | T | T |
| **ParallelTiling** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** | **F** |
| **ParallelThreads** | **F** | - | - | - | - | - | - | **F** | - |

### Phase 2 Key Findings

1. **TERMINAL actions** (nothing composes after): VecSeq, VecPar, ParallelTiling, ParallelThreads
   - These must always be the LAST action in a schedule
   - After these, no linalg op with tag remains for further transformation

2. **COMPOSABLE actions** (most things compose after):
   - **Tiling**: ALL 9 successors OK (most flexible first action)
   - **LoopInterchange**: ALL 9 successors OK (preserves linalg.generic structure)
   - **Packing**: Tiling OK (6D linalg.generic, likely all composable)
   - **Promotion**: 8/9 OK (Promotion→Promotion: Pre=F, can't promote twice)
   - **LoopUnrolling**: 7/8 OK (LoopUnrolling→Packing: Post=F)

3. **Schedule structure**: `{Tiling, LoopInterchange, Packing, Promotion, LoopUnrolling}* → {VecSeq | VecPar | ParallelTiling | ParallelThreads}`

4. **Specific constraints**:
   - Promotion is idempotent-blocked (can only apply once)
   - Packing cannot follow LoopUnrolling
   - Im2col only for conv2d (must precede other actions)

## Phase 2b: Cross-Family Composability Notes

### conv2d-specific
- VecSeq: Post=F on base conv2d (structural incompatibility with windowed ops)
- Im2col: Pre=T, Post=T but 0.33x slower (converts to matmul-like, enabling matmul optimizations)
- ParallelTiling: 22.8x speedup as single action

### pooling-specific
- VecSeq: Post=F on base pooling (same structural issue as conv2d)
- ParallelTiling: 25.1x speedup (best single-action result across all families)

### add/relu (elementwise, all-parallel dims)
- VecSeq: Post=T but slower (0.44x add, 0.54x relu) - vectorization overhead without parallelism
- ParallelTiling: strong (5-6x) but less than contraction/conv families

## Phase 3: Multi-Step Schedule Shapes

### Candidate Schedule Templates (from Phase 2 analysis)

Based on composability findings, the viable multi-step schedule shapes are:

**For matmul/conv2d/pooling (contraction kernels):**
1. `Tiling → ParallelTiling` (tile for cache, then parallelize)
2. `Tiling → VecSeq` (tile for SIMD, then vectorize)
3. `Tiling → Promotion → VecSeq` (tile, promote to stack, vectorize)
4. `Tiling → Promotion → ParallelTiling` (tile, promote, parallelize)
5. `LoopInterchange → ParallelTiling` (reorder, parallelize)

**For add/relu (elementwise):**
1. `ParallelTiling` (single action dominates)
2. `Tiling → ParallelTiling`
3. `Tiling → VecSeq`

**For conv2d with Im2col:**
1. `Im2col → ParallelTiling` (lower to matmul, parallelize)
2. `Im2col → Tiling → VecSeq` (lower, tile, vectorize)

## Phase 4: Schedule Graph and Dependencies Synthesis

### SCHEDULE_GRAPH

Based on all empirical composability data:

```python
SCHEDULE_GRAPH = {
    "matmul": [
        ["ParallelizationTiling"],
        ["Tiling", "ParallelizationTiling"],
        ["Tiling", "VectorizationSequential"],
        ["Tiling", "Promotion", "VectorizationSequential"],
        ["Tiling", "Promotion", "ParallelizationTiling"],
        ["VectorizationSequential"],
        ["ParallelizationThreads"],
        ["Tiling", "ParallelizationThreads"],
        ["Promotion", "ParallelizationTiling"],
        ["LoopInterchange", "ParallelizationTiling"],
    ],
    "conv2d": [
        ["ParallelizationTiling"],
        ["Tiling", "ParallelizationTiling"],
        ["Im2colLowering", "ParallelizationTiling"],
        ["Im2colLowering", "Tiling", "ParallelizationTiling"],
        ["Tiling", "ParallelizationThreads"],
        ["ParallelizationThreads"],
    ],
    "pooling": [
        ["ParallelizationTiling"],
        ["Tiling", "ParallelizationTiling"],
        ["Tiling", "ParallelizationThreads"],
        ["ParallelizationThreads"],
    ],
    "add": [
        ["ParallelizationTiling"],
        ["Tiling", "ParallelizationTiling"],
        ["Tiling", "VectorizationSequential"],
        ["VectorizationSequential"],
        ["ParallelizationThreads"],
    ],
    "relu": [
        ["ParallelizationTiling"],
        ["Tiling", "ParallelizationTiling"],
        ["Tiling", "VectorizationSequential"],
        ["VectorizationSequential"],
        ["ParallelizationThreads"],
    ],
}
```

### ACTION_DEPENDENCIES

Cross-kernel denylist based on empirical failures:

```python
ACTION_DEPENDENCIES = {
    # VecSeq fails Post on conv2d and pooling structured ops
    "VectorizationSequential": ["conv_2d_nchw_fchw", "pooling_nchw_max"],
    # Im2col only works on conv2d
    "Im2colLowering": ["matmul", "pooling_nchw_max", "add", "relu", "generic"],
}
```

### Structural Rules (for behavior masking)
1. Terminal actions (VecSeq, VecPar, ParallelTiling, ParallelThreads) must be LAST
2. Promotion cannot follow Promotion (Pre=F)
3. Packing cannot follow LoopUnrolling (Post=F)
4. Im2col can only precede matmul-compatible actions (converts conv→matmul)
5. Maximum useful schedule depth: 3 actions (diminishing returns beyond)

## Phase 3: Multi-Kernel Execution Benchmarks

### Baselines (3 kernels per family)

| Kernel | MLIR Base (ms) | PyTorch (ms) |
|--------|---------------|-------------|
| matmul_128_128_128 | 2.645 | 0.088 |
| matmul_256_256_256 | 42.144 | 0.136 |
| matmul_512_1024_512 | 521.816 | 0.817 |
| conv_128_32_7_7_32_3_3_3_3 | 13.093 | 0.310 |
| conv_128_32_28_28_32_1_1_28_28 | 59.437 | 0.601 |
| conv_128_32_112_112_64_1_1_56_56 | 864.629 | 14.834 |
| pool_128_128_14_14_7_4_4 | 24.551 | 0.894 |
| pool_128_64_14_14_7_4_4 | 12.275 | 0.446 |
| pool_128_512_14_14_7_4_4 | 98.254 | 3.138 |
| add_112_112_14_15 | 3.379 | 0.551 |
| add_224_120_120_56 | 272.899 | 55.739 |
| add_120_240_28_224 | 266.299 | 55.507 |
| relu_256_32_28_28 | 8.004 | 2.030 |
| relu_256_64_112_112 | 262.776 | 47.059 |
| relu_256_384_56_56 | 392.892 | 70.444 |

### Schedule: ParallelizationTiling [4,4,...,0,0]

| Kernel | ParTile (ms) | MLIR Speedup | vs PyTorch |
|--------|-------------|-------------|-----------|
| matmul_128_128_128 | 0.190 | 13.9x | 0.46x |
| matmul_256_256_256 | 1.717 | 24.5x | 0.08x |
| matmul_512_1024_512 | 19.688 | 26.5x | 0.04x |
| conv_128_32_7_7_32_3_3_3_3 | 0.574 | 22.8x | 0.54x |
| conv_128_32_28_28_32_1_1_28_28 | 5.126 | 11.6x | 0.12x |
| conv_128_32_112_112_64_1_1_56_56 | 40.102 | 21.6x | 0.37x |
| pool_128_128_14_14_7_4_4 | 0.978 | 25.1x | 0.91x |
| pool_128_64_14_14_7_4_4 | 0.545 | 22.5x | 0.82x |
| pool_128_512_14_14_7_4_4 | 3.642 | 27.0x | 0.86x |
| add_112_112_14_15 | 0.639 | 5.29x | 0.86x |
| add_224_120_120_56 | 44.388 | 6.15x | 1.26x |
| add_120_240_28_224 | 43.591 | 6.11x | 1.27x |
| relu_256_32_28_28 | 1.239 | 6.46x | 1.64x |
| relu_256_64_112_112 | 51.086 | 5.15x | 0.92x |
| relu_256_384_56_56 | 73.931 | 5.32x | 0.95x |

### Schedule: ParallelizationThreads [28] (add/relu large tensors)

| Kernel | ParThreads (ms) | MLIR Speedup | vs PyTorch |
|--------|----------------|-------------|-----------|
| add_224_120_120_56 | 42.086 | **6.49x** | **1.32x** |
| add_120_240_28_224 | 35.075 | **7.59x** | **1.58x** |
| relu_256_64_112_112 | 43.457 | **6.05x** | **1.08x** |
| relu_256_384_56_56 | 65.768 | **5.97x** | **1.07x** |

### Schedule: Tiling→ParallelizationTiling (2-step, matmul)

| Kernel | Tile Params | ParTile (ms) | MLIR Speedup | vs ParTile-only |
|--------|-----------|-------------|-------------|----------------|
| matmul_256_256_256 | [32,32,32]→[4,4,0] | 39.758 | 1.06x | 23x worse |
| matmul_512_1024_512 | [64,64,64]→[4,4,0] | 91.121 | 5.73x | 4.6x worse |

### Phase 3 Key Findings

1. **ParallelizationTiling is consistently the best single-action schedule** across ALL families:
   - matmul: 14-27x MLIR speedup (scales with problem size)
   - conv2d: 12-23x
   - pooling: 23-27x
   - add/relu: 5-6x

2. **ParallelizationThreads(28) BEATS PyTorch on large elementwise ops**:
   - add: 1.32-1.58x faster than PyTorch JIT
   - relu: 1.07-1.08x faster than PyTorch JIT
   - Better than ParTile for large tensors (coarser granularity, less overhead)

3. **Tiling→ParTile is WORSE than standalone ParTile**:
   - Outer sequential scf.for loops bottleneck parallelism
   - The RL policy should learn to prefer standalone ParTile

4. **vs PyTorch scaling**: MLIR gap widens for matmul/conv as size increases (PyTorch benefits from BLAS/MKL), but MLIR matches or beats PyTorch on elementwise ops

5. **Best terminal action by family**:
   - matmul/conv/pooling: ParallelizationTiling (fine-grained parallel tiles)
   - add/relu (large): ParallelizationThreads (coarse-grained, less overhead)
   - add/relu (small): ParallelizationTiling (sufficient parallelism from tiling)
