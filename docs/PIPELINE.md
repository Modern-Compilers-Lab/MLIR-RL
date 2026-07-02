# MLIR-RL Pipeline Overview
> Updated: 2026-04-27

---

## Pipeline Summary

```
┌─────────────────────────────────────────────────────────────┐
│  Baselines (once per dataset, shared)                       │
│                                                             │
│  get_base.py        → exec_times/base.json    (MLIR raw)    │
│  get_pytorch_times  → exec_times/pytorch.json (PyTorch)     │
│  split_json.py      → exec_times/base_train.json            │
│                     → exec_times/base_eval.json             │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬─────────────┬─────────────┐
    ▼             ▼             ▼             ▼             ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│baseline│  │  v1    │  │  v2    │  │  v3    │  │  v4    │  │  v4.5  │
│ LSTM   │  │ +HW obs│  │+shaped │  │+transf.│  │+Pad/Pk │  │+Robust │
│6 action│  │ 6 act  │  │ reward │  │ encoder│  │ 9 act  │  │ Isolation
└───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘
    │  train    │  train    │  train    │  train    │  train    │  train
    │  eval     │  eval     │  eval     │  eval     │  eval     │  eval
    └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┘
          │           │           │           │           │
          ▼           ▼           ▼           ▼           ▼
        ┌──────────────────────────────────────────────────┐
        │  dashboard.py        → interactive viz           │
        │  (Head-to-Head tab compares all implementations) │
        └──────────────────────────────────────────────────┘
```

---

## Implementation Versions

| Package | Agent Dir | Key Feature | Actions | Reward | Encoder | Reliability |
|---|---|---|---|---|---|---|
| `rl_autoschedular_v0` | `old_agent` | Baseline | 6 | Sparse terminal | LSTM | In-process |
| `rl_autoschedular_v1` | `v1_agent` | Hardware-aware observation | 6 | Sparse terminal | LSTM + HW concat | In-process |
| `rl_autoschedular_v2` | `v2_agent` | Shaped intermediate reward | 6 | Dense shaped + sparse terminal | LSTM | In-process |
| `rl_autoschedular_v3` | `v3_agent` | Transformer loop-nest encoder | 6 | Sparse terminal | Transformer | In-process |
| `rl_autoschedular_v4` | `v4_agent` | Extended action space | 9 | Sparse terminal | LSTM | In-process |
| `rl_autoschedular_v4_5`| `v4_5_agent`| Robust Integration | 9 | Success-Contingent | Transformer | Isolated Process |

---

## Active Experiment: experiment3

All current development and V4.5 training targets **`results/experiment3`**. 

Standard Config for V4.5:
```json
{
  "implementation": "rl_autoschedular_v4_5",
  "results_dir": "results/experiment3",
  "benchmarks_folder_path": "data/all",
  ...
}
```

**Important:** Each version is a standalone fork. Features do NOT accumulate — v2 does not include v1's hardware, v3 does not include v2's reward, etc.

---

## One-Shot Pipeline

The recommended way to run everything:

```bash
bash scripts/pipeline.sh config/baseline.json
bash scripts/pipeline.sh config/baseline.json "rl_autoschedular_v0,rl_autoschedular_v1,rl_autoschedular_v3"
```

### Config Derivation

The first argument is the **base config** used for shared steps. For per-implementation steps, the script auto-derives `config/<version>.json` from each implementation name:

| Implementation | → Config |
|---|---|
| `rl_autoschedular_v0` | `config/baseline.json` |
| `rl_autoschedular_v1` | `config/v1.json` |
| `rl_autoschedular_v3` | `config/v3.json` |
| unknown / custom | fallback to base config |

This ensures each version gets its own hyperparameters (v3 gets transformer fields, v2 gets reward shaping, etc.).

### Failure Resilience

If any implementation's train or eval submission fails, the pipeline continues with remaining versions. Progress is tracked in:
- `pipeline_logs/pipeline_<timestamp>.log` — full stdout/stderr
- `pipeline_logs/pipeline_<timestamp>.status` — JSON status file (which steps passed/failed)

### Steps executed:

1. `baseline/get_base.sh` (Slurm) — shared MLIR baseline
2. `get_pytorch_times.py` (local) — PyTorch baselines
3. `split_json.py` (local) — stratified train/eval split
4. `train/train.sh` (Slurm) × N implementations (parallel, each with its own config)
5. `eval/eval.sh` (Slurm) × N (dependent on train)

---

## Manual Pipeline (per step)

### Benchmark generation

```
synthetic:  mlir_generators.py → build_benchmark.py → JSON dataset
from model: vision2mlir.py / transformers2mlir.py → _linalg.mlir
            → wrap_mlir.py → code_files/*.mlir
            → get_base.py  → exec_times/base.json
```

### RL training

```
config JSON + exec_times/base_train.json + benchmarks_folder_path/*.mlir
→ train.py → results/<experiment>/<impl>_agent/run_N/{models/, logs/, exec_data.json}
```

### Evaluation

```
config JSON + trained checkpoint (.pt)
→ eval.py → results/<experiment>/<impl>_agent/run_N/logs/eval/{speedup/, exec_time/}
```

### Understanding Train vs. Eval (`train.sh` vs `eval.sh`)

- **In-Training Evaluation (`train.sh`)**: The RL agent periodically pauses training to measure validation progress on an evaluation set. This step tells you if the model is actually learning, and it saves intermediate model checkpoints (e.g., `model_200.pt`) while updating `exec_data.json`.
- **Standalone Verification (`eval.sh`)**: Run *after* training is finished. It relies on the `.pt` checkpoints created by `train.sh` and evaluates each one rigorously against the full `base_eval.json` split (with RL randomness disabled). This populates the final statistics needed for the Dashboard to compare checkpoint-vs-baseline performance.

### Comparison

Comparison is done interactively via the **Dashboard** (see below). The Head-to-Head tab provides per-implementation speedup, win/loss/tie matrix, and Wilcoxon tests.

---

## Results Directory Layout

```
results/<experiment>/
├── exec_times/                         # SHARED across all implementations
│   ├── base.json                       #   All benchmarks, unoptimized MLIR
│   ├── base_train.json                 #   Train split
│   ├── base_eval.json                  #   Eval split
│   └── pytorch.json                    #   PyTorch eager/compile/jit
│
├── old_agent/                          # rl_autoschedular_v0 (baseline)
│   └── run_0/
│       ├── models/model_*.pt           # Checkpoints
│       ├── exec_data.json              # Schedule→exec_time cache
│       └── logs/
│           ├── train/                  # reward, entropy, final_speedup
│           ├── train_ppo/              # policy_loss, value_loss, clip_frac
│           ├── train_value/            # value model loss
│           └── eval/
│               ├── speedup/<bench>     # Per-benchmark speedup per eval step
│               ├── exec_time/<bench>   # Per-benchmark exec time per eval step
│               ├── average_speedup     # Mean speedup across eval benchmarks
│               ├── cumulative_reward   # Cumulative reward per eval batch
│               ├── reward, entropy     # Per-step eval metrics
│               └── final_speedup       # Final speedup per benchmark
│
├── v1_agent/                           # rl_autoschedular_v1
├── v2_agent/                           # rl_autoschedular_v2
├── v3_agent/                           # rl_autoschedular_v3
├── v4_agent/                           # rl_autoschedular_v4
```

---

## Config Files

| File | Implementation | Description |
|---|---|---|
| `config/baseline.json` | `rl_autoschedular` | PPO training config for baseline |
| `config/v1.json` | `rl_autoschedular_v1` | With hardware-aware observation |
| `config/v2.json` | `rl_autoschedular_v2` | With shaped reward |
| `config/v3.json` | `rl_autoschedular_v3` | With transformer encoder |
| `config/v4.json` | `rl_autoschedular_v4` | With extended action space |

Configs no longer need `json_file` or `eval_json_file` — train/eval split paths are auto-derived from `results_dir` + implementation. The `Benchmarks` class first checks for implementation-specific split files (backward compatibility), then falls back to the shared `base_train.json` / `base_eval.json`.

---

## Auto-Derived File Resolution

When `json_file` is not set in the config, the `Benchmarks` class resolves paths like this:

```
Training benchmarks:
  1. results/<exp>/exec_times/old_base_train.json  (if exists, backward compat)
  2. results/<exp>/exec_times/base_train.json       (shared fallback)

Eval benchmarks:
  1. results/<exp>/exec_times/old_base_eval.json
  2. results/<exp>/exec_times/base_eval.json
```

This lets all implementations share the same baseline and split files while preserving backward compatibility.

---

## Dashboard

The dashboard (`dashboard/dashboard.py`) auto-discovers all `*_agent/` directories under the selected experiment and provides:

- **Exec Times** — RL vs MLIR baseline vs PyTorch grouped bar charts
- **Speedup** — CDF, heatmaps, per-benchmark evolution, best-so-far convergence
- **Training** — Reward, entropy, cumulative reward curves
- **Checkpoints** — Exec time evolution across checkpoints
- **Schedules** — Unique schedule count, transform usage, Pareto frontier
- **Head-to-Head** — Pairwise scatter, win/loss/tie counts, Wilcoxon tests
- **Correlation** — Batch size vs speedup, sequence length vs speedup

No additional setup needed — just point it at `results/`:

```bash
cd dashboard
streamlit run dashboard.py
```

---

## Quick Commands Cheat-Sheet

```bash
# Environment
source ~/envs/mlir/bin/activate && set -a && source .env && set +a

# Shared baselines (once)
sbatch scripts/baseline/get_base.sh config/baseline.json
python scripts/get_pytorch_times.py --config config/baseline.json
python scripts/split_json.py --config config/baseline.json

# Train + eval per version (each with its own config)
sbatch scripts/train/train.sh config/baseline.json
sbatch scripts/train/train.sh config/v3.json rl_autoschedular_v3
sbatch scripts/eval/eval.sh config/baseline.json
sbatch scripts/eval/eval.sh config/v3.json rl_autoschedular_v3

# One-shot pipeline (all versions, with auto-derived configs)
bash scripts/pipeline.sh config/baseline.json
bash scripts/pipeline.sh config/baseline.json "rl_autoschedular,rl_autoschedular_v1,rl_autoschedular_v3"

# Monitor pipeline
cat pipeline_logs/pipeline_*.status | python3 -m json.tool

# Compare in dashboard
cd dashboard && streamlit run dashboard.py
```
