# MLIR-RL Paper vs Manuscript — Methodology Alignment

## Hyperparameters

| Parameter | MLIR-RL Paper | Our Manuscript (before) | Our Manuscript (after) |
|---|---|---|---|
| Learning rate | 0.001 | 0.0003 | → 0.001 |
| Training iterations | 10,000 | 2,000 | → 10,000 |
| Benchmarks per trajectory | 64 | 32 | → 64 |
| PPO mini-batch size | 32 | 64 | → 32 |
| PPO clip range (ε) | 0.2 | 0.2 | ✓ same |
| Discount factor (γ) | 1.0 | 1.0 | ✓ same |
| GAE λ | 0.95 | 0.95 | ✓ same |
| PPO epochs | 4 | 4 | ✓ same |
| Entropy coefficient | 0.01 | 0.01 | ✓ same |
| Value loss coefficient | 0.5 | 0.5 | ✓ same |
| Max gradient norm | not specified | 0.5 | 0.5 (our addition) |
| Truncation length | 5 | 5 | ✓ same |

## Methodology

| Aspect | MLIR-RL Paper | Our Manuscript (before) | Our Manuscript (after) |
|---|---|---|---|
| Speedup formula | T_baseline / T_optimized | T_MLIR-Base / T_optimized | → match paper definition explicitly |
| Execution protocol | 5 runs, median | 5 runs, median | ✓ same (just added) |
| Aggregate metric | not explicitly geometric mean (uses per-op speedups) | geometric mean | ✓ (just changed from arithmetic) |
| Baseline definition | "MLIR compiled without loop-level optimizations, -O3 from LLVM" | "MLIR-Base" (undefined) | → add explicit definition |
| PyTorch eval protocol | 10 warm-up + 11 runs, median | not specified | → add |
| Reward | log-speedup (sparse terminal) | log-speedup (sparse terminal) | ✓ same (we also have shaped, paper does not) |

## Our Additions (Not in the Paper)

These are our contributions and should remain in the manuscript:

| Feature | Description |
|---|---|
| Hardware-aware observation | 7-element feature vector (L1/L2/L3 cache, physical/logical cores, SIMD width, clock MHz) |
| Shaped reward | Dense intermediate rewards from static analysis (arithmetic intensity, vectorizability, parallel-loop ratio) |
| Transformer encoder | Replaces LSTM with multi-head self-attention (d_model=256, nhead=8, 3 layers, FFN=1024, dropout=0.1, GELU) |
| Expanded action space | +Pad, Pack, Unroll (3 tile sizes, 3 pad multiples, 3 unroll factors) |
| Dataset | 18 models across 8 categories (vs. paper's synthesized + extracted ops) |
| Multi-hardware evaluation | Train on Bergamo, evaluate on Jubail + Dalma (zero-shot) |
| Training hardware | Bergamo (256-core AMD EPYC 9754) vs. paper's Dalma (28-core Intel Xeon E5-2680 v4) |
