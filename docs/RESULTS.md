# Experimentation Results

## Single-Ops Dataset Experiments

## Overview

Three agents trained and evaluated on `data/single_ops_dataset/all/` (1,569 MLIR files):

| Agent | Architecture | Checkpoints | Best CP | GeoAvg | Failed |
|-------|-------------|-------------|---------|--------|--------|
| V0 | LSTM (baseline) | 90 (100-9000) | 4400 | 1.06x | 1 |
| V4.9-Small | Transformer d=128, 4 heads, 2 layers | 99 (100-9900) | 3800 | 1.21x | 0 |
| V4.9-Large | Transformer d=256, 8 heads, 3 layers | 99 (100-9900) | 3200 | 1.17x | 8 |

**V4.9-Small is the best agent** (highest geo mean, 0 failed benchmarks at best CP).

---

## Key Findings

### 1. V4.9-Small excels at synthetic benchmarks; V4.9-Large excels at model ops

The eval set splits into two tiers:

| Benchmark Type | Count | % | V4.9-Small | V4.9-Large | V0 |
|---------------|-------|---|-----------|-----------|-----|
| **Synthetic** (op+dims names) | 244 | 80% | **1.68x** | 1.12x | **1.65x** |
| **Model-prefixed** (model+op names) | 61 | 20% | 0.33x | **1.41x** | 0.18x |
| **ALL** | 305 | 100% | **1.21x** | 1.17x | 1.06x |

- **Combined evolution plot** → V4.9-Small wins (dominated by synthetic 80%)
- **Per-model plot** → V4.9-Large wins on most individual models
- V4.9-Small overfits to synthetic benchmarks; V4.9-Large generalizes better to real model operations

### 2. Per-model breakdown (best checkpoints)

| Model | Benchs | V0 | V4.9-Small | V4.9-Large |
|-------|--------|-----|-----------|-----------|
| convnext_tiny | 9 | 0.62x | 0.73x | **3.80x** |
| resnet50 | 9 | 0.98x | **3.59x** | 2.97x |
| resnext50 | 4 | 1.47x | 0.61x | 1.41x |
| vit_b_16 | 2 | 2.23x | 1.60x | **3.01x** |
| efficientnet_b0 | 7 | 0.08x | 0.58x | **1.06x** |
| whisper_base | 4 | 0.98x | 1.06x | **10.60x** |
| yolov8m | 3 | 0.46x | 0.32x | **1.74x** |
| gpt2 | 3 | 0.03x | 0.11x | **0.64x** |
| t5 | 2 | 0.03x | 0.02x | **1.09x** |
| albert | 4 | 0.03x | 0.05x | **0.60x** |
| distilbert | 1 | 0.00x | 0.00x | **1.34x** |
| gat | 1 | 0.09x | 0.55x | **0.58x** |
| mobilenet_v3_small | 12 | 0.03x | 0.05x | **0.43x** |

### 3. Per-operation breakdown (best checkpoints)

| Operation | V0 | V4.9-Small | V4.9-Large |
|-----------|-----|-----------|-----------|
| matmul | **14.94x** | 8.29x | 5.91x |
| conv2d | 1.71x | 2.43x | **3.56x** |
| pooling | 0.76x | **0.91x** | 0.55x |
| add | 0.39x | **0.45x** | 0.36x |
| relu | **0.84x** | 0.69x | 0.22x |

### 4. Entropy collapse (all transformer-based agents)

All V4.x agents (V4.6/7/8/9) suffer entropy collapse mid-training:
- V4.9-Small collapses ~CP 6000-7000 (geo mean drops from 1.2x to 0.9x)
- V4.9-Large collapses ~CP 3300-4000 (geo mean drops from 1.17x to 0.6x)
- V0 (LSTM) maintains performance throughout, no collapse

### 5. Larger model does not help

V4.9-Large (d=256, 8 heads, 3 layers) underperforms V4.9-Small (d=128, 4 heads, 2 layers):
- Best geo mean: 1.17x vs 1.21x
- Collapses earlier and more severely
- Requires 32GB RAM for eval (vs 16GB for small)

---

## Train/Eval Split Issue

The `model_family()` function in `scripts/data/split_json.py` used `OP_KEYWORDS` to split benchmark names on `_` and match individual tokens. Compound keywords like `batch_matmul` were split into `batch` then `matmul`, causing model families like `bart` to fragment into `bart`, `bart_batch`, `bart_reduce`, `bart_sub`, etc. This produced an incorrect stratified split where **5 models had 0 eval benchmarks** (bart, bert, gin, llama3_2_1b, vgg16).

**Fix applied**: `model_family()` now first checks `KNOWN_MODELS` list (longest prefix match) before falling back to keyword-based extraction. The corrected split gives all 18 models eval representation.

**Impact on plots**: All existing plots use the original (broken) eval set of 305 benchmarks. The per-model plot shows only 13 of 18 model types. Results should be interpreted with this caveat.

---

## Results Directory Structure

```
results/single_ops_dataset_results/
├── baselines/mlir/
│   ├── base.json              # 1591 total baselines
│   ├── base_train.json        # 1224 train (old split), 1239 (new split)
│   └── base_eval.json         # 305 eval (old split), 312 (new split)
├── v0_agent/
│   ├── eval/                  # 90 checkpoints
│   └── models/                # model_N.pt checkpoints (max 9950)
├── v4_9_small_agent/
│   ├── eval/                  # 99 checkpoints
│   └── models/
└── v4_9_large_agent/
    ├── eval/                  # 99 checkpoints
    └── models/
```

## Plots

```
plots/version_comparison/single_ops_dataset/
├── combined/combined.png      # All 3 agents eval evolution
├── per_operation/             # Speedup by op type (bar chart)
├── per_model/                 # Speedup by model (bar chart)
├── v0/v0.png                  # V0 eval evolution
├── v4_9_small/                # V4.9-Small eval evolution
└── v4_9_large/                # V4.9-Large eval evolution
```

## Configs

```
config/single_ops_dataset/
├── train/
│   ├── v0.json                # V0 training
│   ├── v4_9_small.json        # V4.9-Small training
│   ├── v4_9_large.json        # V4.9-Large training
│   └── v4_6.json .. v4_8.json # Legacy (not used)
└── eval/
    ├── v0_eval.json
    ├── v4_9_small_eval.json
    ├── v4_9_large_eval.json
    └── v4_6_eval.json .. v4_8_eval.json  # Legacy (not used)
```

---

## Ops-and-Blocks Dataset

Merges `single_ops_dataset` (1,569 single ops) with `bench_train/` blocks (7,393 multi-op sequences) into a single benchmark set.

### Structure

```
data/ops_and_blocks/
├── all/         # 8,962 MLIR files (symlinks)
├── train/       # 6,553 (stratified split)
└── eval/        # 1,687 (stratified split)
```

### Multi-level Stratified Split

Uses `scripts/data/split_ops_and_blocks.py` with three-level stratification:

1. **Category**: `single_op` vs `block`  
2. **Model**: extracted from filename prefix (18 known models + synthetic)  
3. **Operation type** (for single ops only): extracted from filename suffix  

**Result**: Every model has at least 1 train + 1 eval benchmark in both categories.

### Split Distribution

| Category | Train | Eval | Eval% |
|----------|-------|------|-------|
| Blocks (all 18 models) | 6,053 | 1,514 | 20.0% |
| Single-op (all 18 models) | 500 | 173 | 25.7% |
| — of which synthetic | 1,188 | 297 | 20.0% |
| **Total** | **6,553** | **1,687** | **20.5%** |

### Baselines

Generated by merging `single_ops_dataset` baselines (1,591) + `new_dataset` baselines (10,985), then filtering to the 8,962 files in the dataset. 740 previously-untimed blocks were re-baselined (58/740 succeeded, rest fail/timeout).

```
results/ops_and_blocks_results/baselines/mlir/
├── base.json        # 8,280 entries (8,240 valid)
├── base_train.json  # 6,553 entries
└── base_eval.json   # 1,687 entries
```

