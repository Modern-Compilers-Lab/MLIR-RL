# MLIR Schedule Exploration Log: All Kernel Families
- Action Version: v53
- Kernels: matmul, conv2d, pooling, add, relu (dataset_ml train split)
- Hardware: Intel Xeon E5-2680 v4 (Broadwell), AVX2, 28 cores
- Date: 2026-05-31

## Baselines

| Kernel | MLIR Base (ms) | PyTorch (ms) |
|--------|---------------|-------------|
| matmul_512_128_512 | 86.84 | 0.17 |
| conv2d_128_128_7_7_48_3_3_3_3 | 80.03 | 0.85 |
| pooling_128_64_14_14_7_4_4 | 12.29 | 0.44 |
| add_112_112_7_28 | 3.03 | 0.41 |
| relu_128_96_15_15 | 3.06 | 0.60 |

## Phase 1: Single Actions

### matmul_512_128_512 (base: 86.84ms)
| # | Action | Parameters | Pre | Post | Time (ms) | Speedup | Speedup to PyTorch |
|---|--------|-----------|-----|------|-----------|---------|-------------------|
| S1 | Tiling | {tile_sizes: [16,16,16]} | T | T | 25.06 | 3.47x | 0.007x |
| S2 | LoopInterchange | {permutation: [1,2,0]} | T | T | 410.08 | 0.21x | 0.0004x |
| S3 | Promotion | {tile_sizes: [16,16,16]} | T | T | 29.25 | 2.97x | 0.006x |
| S4 | SequentialVectorization | {vector_sizes: [4,4,4]} | T | T | 14.36 | 6.05x | 0.012x |
| S5 | ParallelVectorization | {vector_sizes: [4,4,4]} | T | T | 0.64 | 135.2x | 0.26x |
| S6 | Im2colLowering | {} | F | F | N/A | N/A | N/A |
| S7 | TilingParallelization | {tile_sizes: [64,64,0]} | T | T | 4.42 | 19.7x | 0.038x |
| S8 | ThreadParallelization | {num_threads: 16} | T | T | 5.96 | 14.6x | 0.028x |

### conv2d_128_128_7_7_48_3_3_3_3 (base: 80.03ms)
| # | Action | Parameters | Pre | Post | Time (ms) | Speedup | Speedup to PyTorch |
|---|--------|-----------|-----|------|-----------|---------|-------------------|
| S1 | Tiling | {tile_sizes: [16,16,0,0,16,0,0]} | T | T | 72.13 | 1.11x | 0.012x |
| S2 | LoopInterchange | {permutation: [1,0,2,3,4,5,6]} | T | T | - | - | - |
| S3 | Promotion | {tile_sizes: [16,16,0,0,16,0,0]} | T | T | - | - | - |
| S4 | SequentialVectorization | {vector_sizes: [4,4,1,1,4,1,1]} | T | F | N/A | N/A | N/A |
| S5 | ParallelVectorization | {vector_sizes: [4,4,1,1,4,1,1]} | T | F | N/A | N/A | N/A |
| S6 | Im2colLowering | {} | T | T | 236.68 | 0.34x | 0.004x |
| S7 | TilingParallelization | {tile_sizes: [16,16,0,0,0,0,0]} | T | T | 3.43 | 23.3x | 0.25x |
| S8 | ThreadParallelization | {num_threads: 16} | T | T | 5.09 | 15.7x | 0.17x |

### pooling_128_64_14_14_7_4_4 (base: 12.29ms)
| # | Action | Parameters | Pre | Post | Time (ms) | Speedup | Speedup to PyTorch |
|---|--------|-----------|-----|------|-----------|---------|-------------------|
| S1 | TilingParallelization | {tile_sizes: [16,16,4,4,0,0]} | T | T | 0.84 | 14.6x | 0.52x |
| S2 | ParallelVectorization | {vector_sizes: [4,4,4,4,1,1]} | T | F | N/A | N/A | N/A |
| S3 | ThreadParallelization | {num_threads: 16} | T | T | 0.85 | 14.5x | 0.52x |

### add_112_112_7_28 (base: 3.03ms)
| # | Action | Parameters | Pre | Post | Time (ms) | Speedup | Speedup to PyTorch |
|---|--------|-----------|-----|------|-----------|---------|-------------------|
| S1 | Tiling | {tile_sizes: [16,16,0,4]} | T | T | - | - | - |
| S2 | ParallelVectorization | {vector_sizes: [4,4,1,4]} | T | T | 9.61 | 0.32x | 0.043x |
| S3 | TilingParallelization | {tile_sizes: [16,16,0,4]} | T | T | 1.22 | 2.48x | 0.34x |

### relu_128_96_15_15 (base: 3.06ms)
| # | Action | Parameters | Pre | Post | Time (ms) | Speedup | Speedup to PyTorch |
|---|--------|-----------|-----|------|-----------|---------|-------------------|
| S1 | ParallelVectorization | {vector_sizes: [4,4,1,1]} | T | T | 1.09 | 2.81x | 0.55x |
| S2 | TilingParallelization | {tile_sizes: [16,16,0,0]} | T | T | 1.11 | 2.76x | 0.54x |

## Phase 2: Pairwise Compositions (matmul_512_128_512)

| # | A -> B | Params A | Params B | Pre | Post | Time (ms) | Speedup | Speedup to PyTorch |
|---|--------|----------|----------|-----|------|-----------|---------|-------------------|
| P1 | TilingPar → Tiling | [64,64,0] | [16,16,16] | T | T | 1.67 | 52.0x | 0.10x |
| P2 | TilingPar → SeqVec | [64,64,0] | [4,4,4] | T | T | 0.80 | 108.6x | 0.21x |
| P3 | TilingPar → ParVec | [64,64,0] | [4,4,4] | T | T | 0.86 | 101.0x | 0.20x |
| P4 | Tiling → SeqVec | [16,16,16] | [4,4,4] | T | T | 14.36 | 6.05x | 0.012x |
| P5 | LoopInterchange → ParVec | [1,2,0] | [4,4,4] | T | T | 1.13 | 76.8x | 0.15x |
| P6 | SeqVec → Tiling | [4,4,4] | [16,16,16] | T | **F** | N/A | N/A | N/A |
| P7 | TilingPar → TilingPar | [64,64,0] | [16,16,0] | T | T | - | - | - |
| P8 | TilingPar → ThreadPar | [64,64,0] | 4 | T | T | - | - | - |
| P9 | TilingPar → LoopInterchange | [64,64,0] | [2,0,1] | T | T | - | - | - |
| P10 | TilingPar → Promotion | [64,64,0] | [16,16,16] | T | T | - | - | - |

## Phase 3: Triple Compositions (matmul_512_128_512)

| # | Schedule Path | Time (ms) | Speedup | Speedup to PyTorch |
|---|--------------|-----------|---------|-------------------|
| T1 | TilingPar[32,32,0] → Tiling[8,8,16] → ParVec[8,8,16] | 0.93 | 93.4x | 0.18x |
| T2 | TilingPar[32,32,0] → Tiling[8,8,16] → SeqVec[8,8,16] | 0.92 | 94.4x | 0.18x |

## Key Structural Findings

### Terminal Actions
- **SequentialVectorization** and **ParallelVectorization** are TERMINAL on matmul: after vectorization, the tag moves from the linalg op to the surrounding scf.for/forall loop. Subsequent actions that expect a linalg-tagged op fail postcondition.
- Direct **Vectorization fails postcondition on conv2d and pooling** (non-divisible spatial dims with strides).

### Action Applicability
- **Im2colLowering**: Only applicable to conv_2d_nchw_fchw (precondition fails on matmul, pooling, add, relu)
- **LoopInterchange**: Applicable to conv2d and matmul (ops with reduction loops); fails on add/relu (all parallel loops)
- **All parallelization/tiling**: Universally applicable across families

### Composability Rules (matmul)
- TilingPar → {Tiling, SeqVec, ParVec, TilingPar, ThreadPar, Promotion, LoopInterchange} ✓
- ThreadPar → {Tiling} ✓
- Tiling → {SeqVec, ParVec, Tiling, TilingPar} ✓
- LoopInterchange → {ParVec} ✓
- SeqVec/ParVec → anything = FAIL(post) — **TERMINAL**
- Promotion → SeqVec: structural pass but **LLVM lowering fails** (not viable)

### Composability Rules (conv2d)
- TilingPar → {Tiling} ✓
- Im2col → {TilingPar, Tiling} ✓ (converts to matmul-like generic)
- Im2col → TilingPar → {SeqVec, ParVec} ✓ (after im2col, vectorization works)
- Direct Vectorization → FAIL(post) on raw conv2d

### Composability Rules (pooling)
- TilingPar → {Tiling, ThreadPar} ✓
- Vectorization → FAIL(post) always (non-divisible spatial dims)

### Composability Rules (add/relu)
- TilingPar → {ParVec, SeqVec} ✓ (for relu especially)
- ThreadPar works as single step
- No reduction dims → LoopInterchange not applicable

## Performance Summary (Best Schedules)

| Family | Best Schedule Shape | Time (ms) | Speedup over base | vs PyTorch |
|--------|-------------------|-----------|-------------------|-----------|
| matmul | ParVec alone | 0.64 | 135x | 0.26x |
| matmul | TilingPar→SeqVec | 0.80 | 109x | 0.21x |
| matmul | TilingPar→ParVec | 0.86 | 101x | 0.20x |
| matmul | TilingPar→Tiling→SeqVec | 0.92 | 94x | 0.18x |
| conv2d | TilingPar | 3.43 | 23x | 0.25x |
| conv2d | Im2col→TilingPar→SeqVec | 4.40 | 18x | 0.19x* |
| pooling | TilingPar | 0.84 | 15x | 0.52x |
| pooling | ThreadPar | 0.85 | 14x | 0.52x |
| add | TilingPar | 1.22 | 2.5x | 0.34x |
| relu | ParVec | 1.09 | 2.8x | 0.55x |
| relu | TilingPar | 1.11 | 2.8x | 0.54x |

*Conv2d Im2col result measured on representative kernel; varies by shape.

## Phase 4: Multi-Kernel Validation (Parallel Agents)

### Conv2d (3 kernels tested)
| Kernel | Best Shape | Time (ms) | vs Base | vs PyTorch |
|--------|-----------|-----------|---------|-----------|
| K1: 128x128x7x7, F=48, 3x3, s=2 | TilingPar[4,16,0,0,32,0,0] | 2.91 | 27.4x | 0.47x |
| K2: 256x288x7x7, F=48, 3x3, s=2 | TilingPar[8,16,0,0,64,0,0] | 13.66 | 26.5x | 0.34x |
| K3: 128x32x15x15, F=32, 1x1, s=2 | TilingPar[4,32,8,8,0,0,0] | 0.52 | 13.4x | 0.43x |

Conv2d key finding: Im2col→TilingPar→SeqVec enables vectorization (K1: 4.40ms, K2: 23.74ms) but pure TilingPar with reduction-dim tiling is faster.

### Pooling (3 kernels tested)
| Kernel | Best Shape | Time (ms) | vs Base | vs PyTorch |
|--------|-----------|-----------|---------|-----------|
| P1: 128x64x14x14, K=7x7, s=2 | TilingPar[16,16,4,4,0,0] | 0.83 | 14.7x | 0.53x |
| P2: 128x256x150x150, K=1x1, s=2 | **ThreadPar(32)** | 37.65 | 8.8x | **3.68x** |
| P3: 128x48x7x7, K=1x1, s=2 | TilingPar[16,16,4,4,0,0] | 0.12 | 1.1x | **1.05x** |

Pooling key finding: **ThreadPar BEATS PyTorch 3.68x** on large spatial pooling! Vectorization never works on pooling_nchw_max.

### Add (3 kernels tested)
| Kernel | Best Shape | Time (ms) | vs Base | vs PyTorch |
|--------|-----------|-----------|---------|-----------|
| A1: 112x112x7x28 | TilingPar→SeqVec | 1.10 | 2.81x | 0.38x |
| A2: 112x224x28x120 | **ThreadPar(32)** | 18.10 | 6.95x | **1.47x** |
| A3: 56x15x130x56 | TilingPar[8,0,0,8] | 2.47 | 3.54x | 0.89x |

Add key finding: **ThreadPar BEATS PyTorch 1.47x** on medium-large tensors! TilingPar→SeqVec works for small kernels.

### ReLU (3 kernels tested)
| Kernel | Best Shape | Time (ms) | vs Base | vs PyTorch |
|--------|-----------|-----------|---------|-----------|
| R1: 128x96x15x15 | TilingPar[16,16,0,0] | 1.11 | 2.88x | 0.54x |
| R2: 256x64x56x56 | **ThreadPar(32)** | 9.64 | 6.84x | **1.27x** |
| R3: 256x32x112x112 | **ThreadPar(32)** | 20.51 | 6.38x | **1.17x** |

ReLU key finding: **ThreadPar BEATS PyTorch** on large tensors! TilingPar→Vec fails postcondition for large tensors.

## Critical Discovery: ThreadPar Beats PyTorch on Memory-Bound Ops

For memory-bound operations (pooling, add, relu) with large tensors, simple ThreadParallelization(32) consistently **exceeds PyTorch performance** by 1.17x-3.68x. This is likely because:
1. The 28-core Xeon achieves near-peak memory bandwidth with 32 threads
2. PyTorch's overhead (gradient tracking, tensor metadata) hurts on these simple ops
3. Our direct MLIR compilation eliminates framework overhead

## Synthesis Output

### SCHEDULE_GRAPH (written to registry.py)
- **matmul**: 17 schedule paths (including 5 terminal vectorization paths)
- **conv2d**: 11 schedule paths (Im2col pipeline + direct parallelization)
- **pooling**: 5 schedule paths (parallelization only)
- **add**: 6 schedule paths
- **relu**: 7 schedule paths

### ACTION_DEPENDENCIES (written to registry.py)
- **matmul**: [Im2colLowering] — precondition always fails
- **conv2d**: [] — all actions valid in some context
- **pooling**: [Im2colLowering, SequentialVectorization, ParallelVectorization]
- **add**: [Im2colLowering, LoopInterchange]
- **relu**: [Im2colLowering, LoopInterchange]
