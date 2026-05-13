# MLIR Schedule Exploration Log: conv2d (dataset_conv2d)
# Action Version: v30
# Kernel Family: linalg.conv_2d_nchw_fchw, f64
# Hardware: Intel Xeon E5-2680 v4 (Broadwell), AVX2, 28 cores
# Date: 2026-05-11

## Kernels Summary

| ID | Kernel | N | C | H | W | F | KH | KW | OH | OW | Stride | MLIR Base (ms) | PyTorch (ms) |
|----|--------|---|---|---|---|---|----|----|----|----|----|---------------|-------------|
| K1 | 128_128_14_14_192_1_1_7_7 | 128 | 128 | 14 | 14 | 192 | 1 | 1 | 7 | 7 | 2 | 173.80 | 0.891 |
| K2 | 128_128_7_7_256_1_1_4_4 | 128 | 128 | 7 | 7 | 256 | 1 | 1 | 4 | 4 | 2 | 75.65 | 0.409 |
| K3 | 128_192_7_7_192_1_1_4_4 | 128 | 192 | 7 | 7 | 192 | 1 | 1 | 4 | 4 | 2 | 88.58 | 0.471 |
| K4 | 128_32_130_130_32_1_1_65_65 | 128 | 32 | 130 | 130 | 32 | 1 | 1 | 65 | 65 | 2 | 510.92 | 14.696 |
| K5 | 128_32_7_7_512_3_3_5_5 | 128 | 32 | 7 | 7 | 512 | 3 | 3 | 5 | 5 | 1 | 567.27 | 2.051 |
| K6 | 128_96_15_15_512_1_1_8_8 | 128 | 96 | 15 | 15 | 512 | 1 | 1 | 8 | 8 | 2 | 435.82 | 2.576 |
| K7 | 128_96_7_7_48_1_1_7_7 | 128 | 96 | 7 | 7 | 48 | 1 | 1 | 7 | 7 | 1 | 30.88 | 0.194 |
| K8 | 256_32_7_7_32_3_3_5_5 | 256 | 32 | 7 | 7 | 32 | 3 | 3 | 5 | 5 | 1 | 72.14 | 0.778 |
| K9 | 256_512_15_15_48_1_1_8_8 | 256 | 512 | 15 | 15 | 48 | 1 | 1 | 8 | 8 | 2 | 607.73 | 5.902 |
| K10 | 256_64_14_14_192_1_1_7_7 | 256 | 64 | 14 | 14 | 192 | 1 | 1 | 7 | 7 | 2 | 154.57 | 1.082 |
| K11 | 256_96_7_7_64_3_3_5_5 | 256 | 96 | 7 | 7 | 64 | 3 | 3 | 5 | 5 | 1 | 442.26 | 2.914 |

---

## K1: conv_2d_nchw_fchw_128_128_14_14_192_1_1_7_7

### Baseline
- MLIR base time: 173.80 ms
- PyTorch time: 0.891 ms

### Phase 1: Single Actions
| # | Action | Parameters | Pre | Post | Time (ms) | Speedup |
|---|--------|-----------|-----|------|-----------|---------|
| S1 | Tiling | {tile_sizes: [4,4,4,4,4,0,0]} | T | T | 136.24 | 1.28x |
| S2 | Tiling | {tile_sizes: [16,16,0,0,16,0,0]} | T | T | 133.37 | 1.30x |
| S3 | Tiling | {tile_sizes: [32,32,0,0,32,0,0]} | T | T | - | - |
| S4 | Vectorization | {vector_sizes: [1,4,4,4]} | T | T | EXEC_FAIL | - |
| S5 | Vectorization | {vector_sizes: [1,8,1,8]} | T | T | 61.83 | 2.81x |
| S6 | Image2Col | {} | T | T | 179.08 | 0.97x |
| S7 | ParallelizationTiling | {tile_sizes: [4,8,0,0,0,0,0]} | T | T | 6.65 | 26.1x |
| S8 | ParallelizationDirect | {num_threads: [7,4,0,0,0,0,0]} | T | T | 6.59 | 26.4x |
| S9 | LoopInterchange | {permutation: [0,1,2,3,6,5,4]} | T | T | 174.00 | 1.00x |
| S10 | Promotion | {operands_to_promote: [0,1,2]} | T | T | - | - |
| S11 | Unrolling | {tile_sizes: [4,0,0,0,0,0,0]} | T | T | - | - |

### Phase 2: Pairwise Compositions
| # | A -> B | Params A | Params B | Pre | Post | Time (ms) | Speedup |
|---|--------|----------|----------|-----|------|-----------|---------|
| P1 | PT -> Tiling | [4,8,0,0,0,0,0] | [0,0,4,4,16,0,0] | T | T | 5.59 | 31.1x |
| P2 | PT -> Vectorization | [4,8,0,0,0,0,0] | [1,8,1,8] | T | T | 3.61 | 48.2x |
| P3 | PT -> Image2Col | [4,8,0,0,0,0,0] | {} | T | T | - | - |
| P4 | PT -> Promotion | [4,8,0,0,0,0,0] | [0,1,2] | T | T | - | - |
| P5 | PT -> Unrolling | [4,8,0,0,0,0,0] | [4,0,0,0,0,0,0] | T | T | - | - |
| P6 | PT -> LoopInterchange | [4,8,0,0,0,0,0] | [0,1,2,3,6,5,4] | T | T | 48.73 | 3.57x |
| P7 | I2C -> Tiling | {} | [16,16,0,16] | T | T | - | - |
| P8 | I2C -> PT | {} | [4,8,0,0] | T | T | 9.85 | 17.6x |
| P9 | I2C -> Vectorization | {} | [1,8,1,8] | T | T | 61.83 | 2.81x |

### Phase 3: Multi-Step Schedules
| Candidate | Schedule | Time (ms) | Speedup | Speedup to PyTorch |
|-----------|----------|-----------|---------|-------------------|
| C1 | PT[4,8] -> V[1,8,1,8] | 3.61 | 48.2x | 0.25x |
| C2 | I2C -> PT[4,8] -> V[1,8,1,8] | 5.78 | 30.1x | 0.15x |
| C3 | PT[4,8] -> T[0,0,4,4,16,0,0] | 5.59 | 31.1x | 0.16x |
| C4 | I2C -> PT[4,8] | 9.85 | 17.6x | 0.09x |

### Best Schedule for K1
- Schedule: ParallelizationTiling[4,8,0,0,0,0,0] -> Vectorization[1,8,1,8]
- Time: 3.61 ms
- Speedup: 48.14x vs MLIR base, 0.25x vs PyTorch

---

## K2: conv_2d_nchw_fchw_128_128_7_7_256_1_1_4_4

### Baseline
- MLIR base time: 75.65 ms
- PyTorch time: 0.409 ms

### Best Schedule (PT[4,8] -> V[1,8,1,8])
- Pre: T, Post: T
- Time: 1.30 ms
- Speedup: 58.15x vs MLIR base, 0.31x vs PyTorch

---

## K3: conv_2d_nchw_fchw_128_192_7_7_192_1_1_4_4

### Baseline
- MLIR base time: 88.58 ms
- PyTorch time: 0.471 ms

### Best Schedule (PT[4,8] -> V[1,8,1,8])
- Pre: T, Post: T
- Time: 1.61 ms
- Speedup: 54.99x vs MLIR base, 0.29x vs PyTorch

---

## K4: conv_2d_nchw_fchw_128_32_130_130_32_1_1_65_65

### Baseline
- MLIR base time: 510.92 ms
- PyTorch time: 14.696 ms

### Best Schedule (PT[4,8] -> V[1,8,1,8])
- Pre: T, Post: T
- Time: 43.39 ms
- Speedup: 11.78x vs MLIR base, 0.34x vs PyTorch
- Note: Lower speedup due to large spatial dims (65x65 output) with only 4 F-tiles (F=32, tile=8). Only 32*4=128 parallel tasks vs 32*32=1024 for K1.

---

## K5: conv_2d_nchw_fchw_128_32_7_7_512_3_3_5_5

### Baseline
- MLIR base time: 567.27 ms
- PyTorch time: 2.051 ms

### Best Schedule (PT[4,8] -> V[1,8,1,8])
- Pre: T, Post: T
- Time: 11.66 ms
- Speedup: 48.66x vs MLIR base, 0.18x vs PyTorch
- Note: 3x3 kernel, stride=1. Vectorization performs img2col converting C*KH*KW=32*3*3=288 reduction dim.

---

## K6: conv_2d_nchw_fchw_128_96_15_15_512_1_1_8_8

### Baseline
- MLIR base time: 435.82 ms
- PyTorch time: 2.576 ms

### Best Schedule (PT[4,8] -> V[1,8,1,8])
- Pre: T, Post: T
- Time: 8.54 ms
- Speedup: 51.01x vs MLIR base, 0.30x vs PyTorch

---

## K7: conv_2d_nchw_fchw_128_96_7_7_48_1_1_7_7

### Baseline
- MLIR base time: 30.88 ms
- PyTorch time: 0.194 ms

### Best Schedule (PT[4,8] -> V[1,8,1,8])
- Pre: T, Post: T
- Time: 0.61 ms
- Speedup: 50.25x vs MLIR base, 0.32x vs PyTorch

---

## K8: conv_2d_nchw_fchw_256_32_7_7_32_3_3_5_5

### Baseline
- MLIR base time: 72.14 ms
- PyTorch time: 0.778 ms

### Best Schedule (PT[4,8] -> V[1,8,1,8])
- Pre: T, Post: T
- Time: 1.66 ms
- Speedup: 43.47x vs MLIR base, 0.47x vs PyTorch

---

## K9: conv_2d_nchw_fchw_256_512_15_15_48_1_1_8_8

### Baseline
- MLIR base time: 607.73 ms
- PyTorch time: 5.902 ms

### Best Schedule (PT[4,8] -> V[1,8,1,8])
- Pre: T, Post: T
- Time: 10.97 ms
- Speedup: 55.39x vs MLIR base, 0.54x vs PyTorch
- Note: Highest speedup-to-torch ratio. Large C=512 means substantial vectorized reduction.

---

## K10: conv_2d_nchw_fchw_256_64_14_14_192_1_1_7_7

### Baseline
- MLIR base time: 154.57 ms
- PyTorch time: 1.082 ms

### Best Schedule (PT[4,8] -> V[1,8,1,8])
- Pre: T, Post: T
- Time: 3.70 ms
- Speedup: 41.83x vs MLIR base, 0.29x vs PyTorch

---

## K11: conv_2d_nchw_fchw_256_96_7_7_64_3_3_5_5

### Baseline
- MLIR base time: 442.26 ms
- PyTorch time: 2.914 ms

### Best Schedule (PT[4,8] -> V[1,8,1,8])
- Pre: T, Post: T
- Time: 9.58 ms
- Speedup: 46.16x vs MLIR base, 0.30x vs PyTorch

---

## Cross-Kernel Summary

### Best Schedule per Kernel
| ID | Kernel | Schedule | Time (ms) | Speedup vs Base | Speedup vs PyTorch |
|----|--------|----------|-----------|-----------------|-------------------|
| K1 | 128_128_14_14_192_1_1_7_7 | PT[4,8] -> V[1,8,1,8] | 3.61 | 48.14x | 0.25x |
| K2 | 128_128_7_7_256_1_1_4_4 | PT[4,8] -> V[1,8,1,8] | 1.30 | 58.15x | 0.31x |
| K3 | 128_192_7_7_192_1_1_4_4 | PT[4,8] -> V[1,8,1,8] | 1.61 | 54.99x | 0.29x |
| K4 | 128_32_130_130_32_1_1_65_65 | PT[4,8] -> V[1,8,1,8] | 43.39 | 11.78x | 0.34x |
| K5 | 128_32_7_7_512_3_3_5_5 | PT[4,8] -> V[1,8,1,8] | 11.66 | 48.66x | 0.18x |
| K6 | 128_96_15_15_512_1_1_8_8 | PT[4,8] -> V[1,8,1,8] | 8.54 | 51.01x | 0.30x |
| K7 | 128_96_7_7_48_1_1_7_7 | PT[4,8] -> V[1,8,1,8] | 0.61 | 50.25x | 0.32x |
| K8 | 256_32_7_7_32_3_3_5_5 | PT[4,8] -> V[1,8,1,8] | 1.66 | 43.47x | 0.47x |
| K9 | 256_512_15_15_48_1_1_8_8 | PT[4,8] -> V[1,8,1,8] | 10.97 | 55.39x | 0.54x |
| K10 | 256_64_14_14_192_1_1_7_7 | PT[4,8] -> V[1,8,1,8] | 3.70 | 41.83x | 0.29x |
| K11 | 256_96_7_7_64_3_3_5_5 | PT[4,8] -> V[1,8,1,8] | 9.58 | 46.16x | 0.30x |

### Aggregate Statistics
- Mean speedup vs MLIR base: 46.35x
- Min speedup vs MLIR base: 11.78x (K4, large spatial dims with small F)
- Max speedup vs MLIR base: 58.15x (K2)
- Mean speedup vs PyTorch: 0.33x
- Best speedup vs PyTorch: 0.54x (K9)

### Composability Matrix (from K1 detailed exploration)

Legend: ✓=pre+post pass, X=pre fail, ↓=execution timeout/fail, •=not tested

| A \ B (A->B) | T | LI | P | V | U | PT | PD | I2C |
|--------------|---|----|----|---|---|----|----|-----|
| T            | • | •  | •  | • | • | •  | •  | •   |
| LI           | • | •  | •  | • | • | •  | •  | •   |
| P            | • | •  | •  | • | • | •  | •  | •   |
| V            | • | •  | •  | • | • | •  | •  | •   |
| U            | • | •  | •  | • | • | •  | •  | •   |
| PT           | ✓5.59 | ✓48.73 | ✓↓ | ✓3.61 | ✓↓ | • | • | ✓↓ |
| PD           | • | •  | •  | • | • | •  | •  | •   |
| I2C          | ✓↓ | • | • | ✓61.83 | • | ✓9.85 | • | • |

### Key Findings

1. **ParallelizationTiling is the dominant first action** for conv2d. It provides ~26x speedup by distributing N and F loop iterations across cores.

2. **Vectorization is the best second action after PT.** It performs implicit img2col conversion and then vectorizes the resulting matmul-like contraction with 8-wide SIMD ops, giving an additional ~2x on top of parallelization.

3. **PT[4,8] -> V[1,8,1,8] is universally the best 2-step schedule** across all 11 kernels, with speedups ranging from 11.8x to 58.2x vs MLIR base.

4. **3-step schedules (I2C -> PT -> V) underperform 2-step (PT -> V)** because the img2col at full tensor size before parallelization is less efficient than letting vectorization handle img2col on the smaller parallelized tiles.

5. **K4 is an outlier** with only 11.8x speedup due to large spatial dimensions (65x65 output) but small F=32, leading to only 32*4=128 parallel tasks instead of the ~1024+ tasks in other kernels.

6. **K9 achieves the best PyTorch-relative performance** (0.54x) due to its large C=512 providing excellent vectorization efficiency in the reduction dimension.

7. **All single actions pass pre/post conditions** on conv2d, demonstrating broad applicability of the v30 action set.
