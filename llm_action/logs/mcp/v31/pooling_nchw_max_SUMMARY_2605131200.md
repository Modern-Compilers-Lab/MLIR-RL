# MLIR Schedule Exploration Summary: pooling_nchw_max Family
# Action Version: v31
# Hardware: Intel Xeon E5-2680 v4 (Broadwell), AVX2, 28 cores
# Date: 2026-05-13

## Cross-Kernel Results

Best schedule for all kernels: **ParallelizationTiling {tile_sizes: [4,0,0,0,0,0]}**

| Kernel | MLIR Base (ms) | PyTorch (ms) | Best Time (ms) | Best Schedule | Speedup vs MLIR | Speedup vs PyTorch |
|--------|---------------|-------------|----------------|---------------|-----------------|-------------------|
| K1 128_128_112_112_1_56_56 | 88.060 | 40.295 | 10.27 | PT[4,0,0,0,0,0] | 8.57x | 3.92x |
| K2 128_128_14_14_3_6_6 | 9.166 | 0.653 | 0.563 | PT[4,0,0,0,0,0] | 16.28x | 1.16x |
| K3 128_288_112_112_1_56_56 | 196.995 | 88.192 | 23.50 | PT[4,0,0,0,0,0] | 8.38x | 3.75x |
| K4 128_288_7_7_1_4_4 | 0.805 | 0.379 | 0.373 | PT[4,0,0,0,0,0] | 2.16x | 1.02x |
| K5 128_48_228_228_3_113_113 | 764.423 | 85.636 | 48.11 | PT[4,0,0,0,0,0] | 15.89x | 1.78x |
| K6 256_128_112_112_3_55_55 | 976.513 | 107.985 | 46.37 | PT[4,0,0,0,0,0] | 21.06x | 2.33x |
| K7 256_240_56_56_3_27_27 | 444.205 | 48.988 | 21.11 | PT[4,0,0,0,0,0] | 21.04x | 2.32x |
| K8 256_256_15_15_3_7_7 | 52.214 | 3.345 | 1.977 | PT[4,0,0,0,0,0] | 26.41x | 1.69x |
| K9 256_32_14_14_7_4_4 | 12.280 | 0.485 | 0.665 | PT[4,0,0,0,0,0] | 18.47x | 0.73x |
| K10 256_384_14_14_1_7_7 | 9.328 | 4.456 | 2.197 | PT[4,0,0,0,0,0] | 4.25x | 2.03x |
| K11 256_48_228_228_1_114_114 | 282.358 | 116.847 | 32.34 | PT[4,0,0,0,0,0] | 8.73x | 3.61x |

## Aggregate Statistics
- Mean speedup vs MLIR base: **13.75x**
- Mean speedup vs PyTorch: **2.21x**
- Kernels beating PyTorch: **10/11** (all except K9)
- Best speedup vs PyTorch: **3.92x** (K1)
- Worst speedup vs PyTorch: **0.73x** (K9 - small kernel with 7x7 kernel window)

## Composability Matrix
Tested on K4 (small kernel for fast iteration). All 36/36 ordered action pairs compose successfully.
**No structural block edges exist for pooling_nchw_max.**

|          | -> T | -> V | -> LI | -> U | -> PT | -> PD |
|----------|------|------|-------|------|-------|-------|
| After T  | OK   | OK   | OK    | OK   | OK    | OK    |
| After V  | OK   | OK   | OK    | OK   | OK    | OK    |
| After LI | OK   | OK   | OK    | OK   | OK    | OK    |
| After U  | OK   | OK   | OK    | OK   | OK    | OK    |
| After PT | OK   | OK   | OK    | OK   | OK    | OK    |
| After PD | OK   | OK   | OK    | OK   | OK    | OK    |

## Multi-Step Schedules
Tested on K1 and large kernels (K5, K6):
- PT -> T: slower than PT alone (15.33ms vs 10.27ms on K1; 66.83ms vs 46.37ms on K6; 61.86ms vs 48.11ms on K5)
- PT -> V: slower than PT alone (17.04ms vs 10.27ms on K1)
- Conclusion: **Single-action PT is the optimal strategy for pooling_nchw_max**

## ACTION_DEPENDENCIES for pooling_nchw_max
```python
# For pooling_nchw_max: no block edges
ACTION_DEPENDENCIES: dict[str, list[str]] = {}
```

## Key Findings

1. **ParallelizationTiling dominates**: PT[4,0,0,0,0,0] is the best schedule for ALL 11 kernels. The batch dimension (N) provides ample parallelism.

2. **No composability constraints**: Unlike the existing registry which claims V blocks LI and V, empirical testing shows ALL pairs compose for pooling_nchw_max. The existing `ACTION_DEPENDENCIES` in registry.py is incorrect for this op family.

3. **Multi-step schedules hurt**: Adding tiling or vectorization after parallelization increases overhead without benefit. The pooling operation is memory-bound, and tiling just adds loop overhead.

4. **PT[4,0,0,0,0,0] scales well**: For N=128, creates 32 tasks; for N=256, creates 64 tasks. Both exceed the 28-core count, providing good load balancing.

5. **Small-spatial kernels benefit less from PT**: K4 (7x7 spatial, 1.02x vs PyTorch) and K9 (14x14 spatial with 7x7 kernel, 0.73x vs PyTorch) show diminishing returns because the per-task work is too small for parallelization overhead.

6. **Tiling and vectorization alone are harmful**: They increase execution time by 2-3x due to loop overhead and reduced compiler optimization opportunities.
