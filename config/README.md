# config/ — MLIR-RL Configuration

Two datasets, each with purpose-specific subfolders.

## Directory Layout

```
config/
├── old_dataset/              # Original experiment configs (V0–V4.5, ablations, multi-cluster)
│   ├── train/                (8)  Version training configs
│   ├── ablation/             (6)  Ablation study configs
│   ├── ablation_full_model/  (3)  Ablation full-model variants
│   ├── eval/                 (15) Per-checkpoint Dalma eval
│   ├── eval_bergamo/         (14) Per-checkpoint Bergamo eval
│   ├── full_model/           (1)  Full-model optimization
│   ├── test/                 (3)  Sanity configs
│   └── misc/                 (3)  One-off special-purpose
│
├── new_dataset/              # New dataset experiments (18 models, 11,770 blocks)
│   ├── train/                (5)  Bergamo training configs
│   ├── eval/                 (5)  Bergamo eval (overall perf + ablation)
│   ├── eval_dalma/           (2)  Multi-hardware: Dalma
│   ├── eval_jubail/          (2)  Multi-hardware: Jubail
│   ├── full_model/           (2)  Full-model eval (all ops)
│   └── test/                 (1)  Sanity
│
└── README.md
```

## `old_dataset/` — Original Experiments

Used by all existing shell scripts. Auto-selection: `config/old_dataset/train/${VERSION}.json`.

| Subfolder | Files | Purpose |
|-----------|-------|---------|
| `train/` | 8 | baseline, v1, v2, v2_5, v3, v4, v4_5, train1 |
| `ablation/` | 6 | v45_no_hw, v45_no_shaped_reward, v45_no_transformer (block-based) |
| `ablation_full_model/` | 3 | Full-model ablation variants |
| `eval/` | 15 | Per-checkpoint eval (700–1999) on Dalma |
| `eval_bergamo/` | 14 | Per-checkpoint eval (800–1999) on Bergamo |
| `full_model/` | 1 | full_model_optim.json |
| `test/` | 3 | test, v3_sanity, v4_sanity |
| `misc/` | 3 | albert, example, hardware_eval |

## `new_dataset/` — New Dataset Experiments

18-model dataset (9,407 train / 2,363 eval / 952 eval_full). All training on Bergamo.

### `train/` — Training Configs (5)

| Config | Impl | Features |
|--------|------|----------|
| `v0.json` | `rl_autoschedular_v0` | Baseline LSTM policy |
| `v4_5.json` | `rl_autoschedular_v4_5` | HW-aware + shaped reward + transformer |
| `v45_no_hw.json` | `rl_autoschedular_v45_no_hw` | Ablation: no hardware-awareness |
| `v45_no_shaped_reward.json` | `rl_autoschedular_v45_no_shaped_reward` | Ablation: no shaped reward |
| `v45_no_transformer.json` | `rl_autoschedular_v45_no_transformer` | Ablation: LSTM instead of transformer |

All point to: `benchmarks_folder_path: data/new_dataset/all/train`
Baseline: `results/new_dataset_results/baselines/exec_times/train_base.json`

### `eval/` — Bergamo Eval (5)

5-run median eval on `data/new_dataset/all/eval/` (2,363 files). Results → `results/new_dataset_results/`.

### `eval_dalma/` + `eval_jubail/` — Multi-Hardware (4)

Cross-hardware eval: V4.5 (with HW) and V45-no-HW (without) on Dalma + Jubail.

### `full_model/` — Full Model Eval (2)

Eval V0 + V4.5 on `data/new_dataset/all/eval_full/` (952 files). Post-process: sum per-model times, geometric mean speedup.

### `test/` — Sanity (1)

Minimal 5-bench, 5-iter test run.

## Baseline Pipeline

```bash
# Generate baseline timing JSONs
python scripts/baseline/get_base.py --config config/new_dataset/train/v4_5.json
# → results/new_dataset_results/baselines/exec_times/train_base.json
# Repeat for eval and eval_full directories
```

## Shell Script Usage

All existing shell scripts point to `config/old_dataset/`. For new dataset:
```bash
sbatch scripts/train/train.sh config/new_dataset/train/v4_5.json
sbatch scripts/eval/eval.sh config/new_dataset/eval/v4_5_eval.json
```
