# MLIR-RL Training Guide

## Setup

```bash
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
```

---

## Config Files

Each implementation version has a dedicated config in `config/`:

| Config | Implementation | Key Feature |
|---|---|---|
| `config/baseline.json` | `rl_autoschedular_v0` | LSTM encoder, 6 actions, sparse reward |
| `config/v1.json` | `rl_autoschedular_v1` | Hardware-aware observation (cache sizes, cores, SIMD) |
| `config/v2.json` | `rl_autoschedular_v2` | Shaped reward (arithmetic intensity, vectorizability, parallelism) |
| `config/v3.json` | `rl_autoschedular_v3` | Transformer loop-nest encoder |
| `config/v4.json` | `rl_autoschedular_v4` | Extended action space (Pad, Pack, Unroll) |

Configs **no longer need** `json_file` or `eval_json_file` — these are now auto-derived from `results_dir` and the implementation. Configs share the same `results_dir` so all versions use the same baselines and train/eval splits.

> For all available fields see `utils/config.py`.

---

## Select Implementation

Implementation resolution order (from `utils/implementation.py`):
1. `AUTOSCHEDULER_IMPL` env var
2. Config `"implementation"` field

Set it per shell or pass it as an override to Slurm scripts:

```bash
export AUTOSCHEDULER_IMPL=rl_autoschedular_v3
```

For Slurm wrappers, pass implementation as the second argument:

```bash
sbatch scripts/train/train.sh config/v3.json rl_autoschedular_v3
sbatch scripts/eval/eval.sh config/v3.json rl_autoschedular_v3
```

---

## Full Pipeline

### Option A — One-shot Pipeline (recommended)

```bash
bash scripts/pipeline.sh config/baseline.json
bash scripts/pipeline.sh config/baseline.json "rl_autoschedular_v0,rl_autoschedular_v1,rl_autoschedular_v3"
```

This submits all steps to Slurm with proper dependencies. Progress is tracked in `pipeline_logs/pipeline_<timestamp>.log` and `pipeline_logs/pipeline_<timestamp>.status`.

**How it works:** The first config argument (`config/baseline.json`) is used for shared steps (MLIR baseline, PyTorch baselines, train/eval split). For per-implementation steps, the script auto-derives `config/<version>.json` from each implementation name:

| Implementation | Derived Config |
|---|---|
| `rl_autoschedular_v0` | `config/baseline.json` |
| `rl_autoschedular_v1` | `config/v1.json` |
| `rl_autoschedular_v3` | `config/v3.json` |
| unknown/custom | fallback to base config |

This ensures each version gets its own hyperparameters (transformer fields for v3, reward shaping for v2, etc.).

**Failure resilience:** If any implementation's train or eval submission fails, the pipeline continues with the remaining versions. The status file records which steps succeeded or failed.

### Option B — Manual (step by step)

### 1. MLIR Baseline (once per dataset, shared across all implementations)

Runs each `.mlir` file unoptimized and records execution time to `exec_times/base.json`:

```bash
# Slurm:
sbatch scripts/baseline/get_base.sh config/baseline.json

# Local:
python scripts/get_base.py --config config/baseline.json
# → results/my_experiment/exec_times/base.json
```

### 2. PyTorch Baselines (once per dataset)

Measures eager, `torch.compile`, and `torch.jit` for each benchmark:

```bash
python scripts/get_pytorch_times.py --config config/baseline.json
# → results/my_experiment/exec_times/pytorch.json
```

### 3. Train/Eval Split (once per dataset)

Stratified split by model family, uses the shared `base.json`:

```bash
python scripts/split_json.py --config config/baseline.json
# reads:  results/my_experiment/exec_times/base.json
# writes: results/my_experiment/exec_times/base_train.json
#         results/my_experiment/exec_times/base_eval.json
```

### 4. Train (per implementation)

```bash
# Slurm:
sbatch scripts/train/train.sh config/baseline.json           # baseline
sbatch scripts/train/train.sh config/v1.json rl_autoschedular_v1
sbatch scripts/train/train.sh config/v3.json rl_autoschedular_v3

# Local:
export CONFIG_FILE_PATH=config/v3.json
export AUTOSCHEDULER_IMPL=rl_autoschedular_v3
python scripts/train.py
```

Checkpoints saved every 5 iterations to:
- `results/my_experiment/old_agent/run_N/models/model_<step>.pt` (baseline)
- `results/my_experiment/v1_agent/run_N/models/model_<step>.pt` (v1)
- `results/my_experiment/v3_agent/run_N/models/model_<step>.pt` (v3)

### 5. Evaluate (per implementation)

```bash
# Slurm — auto-detects latest run_N with checkpoints:
sbatch scripts/eval/eval.sh config/baseline.json
sbatch scripts/eval/eval.sh config/v1.json rl_autoschedular_v1

# Local:
export CONFIG_FILE_PATH=config/v3.json
export AUTOSCHEDULER_IMPL=rl_autoschedular_v3
python scripts/eval.py
```

Results written under:
- `results/my_experiment/<impl>_agent/run_N/logs/eval/`

### 6. Dashboard

```bash
cd dashboard
streamlit run dashboard.py --server.fileWatcherType none
```

The dashboard auto-discovers all `*_agent/` directories and their eval data, providing side-by-side comparison across implementations (exec times, speedup, training curves, head-to-head stats).

---

## Version-Specific Config Fields

### v1 — Hardware-Aware Observation

| Field | Default | Description |
|---|---|---|
| `hardware_auto_detect` | `true` | Auto-detect from `/sys` and `/proc` |
| `hardware_l1_kb` | `0.0` | L1 cache size (0 = auto) |
| `hardware_l2_kb` | `0.0` | L2 cache size (0 = auto) |
| `hardware_l3_kb` | `0.0` | L3 cache size (0 = auto) |
| `hardware_physical_cores` | `0` | Physical core count (0 = auto) |
| `hardware_logical_cores` | `0` | Logical core count (0 = auto) |
| `hardware_simd_width` | `0` | SIMD width in bits (0 = auto) |
| `hardware_clock_mhz` | `0.0` | Clock frequency in MHz (0 = auto) |

Keep `hardware_auto_detect: true` on a single machine. Set manual values only for cross-hardware experiments.

### v2 — Shaped Reward

| Field | Default | Description |
|---|---|---|
| `reward_shaping_enabled` | `true` | Enable dense intermediate reward |
| `reward_shaping_scale` | `1.0` | Global multiplier for reward delta |
| `reward_shaping_clip` | `2.0` | Absolute clip on shaped reward term |
| `reward_shaping_weight_ai` | `1.0` | Weight for arithmetic intensity |
| `reward_shaping_weight_vectorizable` | `0.1` | Weight for vectorizability |
| `reward_shaping_weight_parallel` | `0.1` | Weight for parallel-loop ratio |
| `reward_shaping_vectorization_bonus` | `0.2` | Bonus for vectorization action |

### v3 — Transformer Encoder

| Field | Default | Description |
|---|---|---|
| `transformer_d_model` | `256` | Token embedding hidden size |
| `transformer_nhead` | `8` | Attention heads |
| `transformer_num_layers` | `3` | Encoder layers |
| `transformer_ffn_dim` | `1024` | Feed-forward hidden dim |
| `transformer_dropout` | `0.1` | Dropout rate |
| `transformer_activation` | `'gelu'` | `'relu'` or `'gelu'` |
| `transformer_pooling` | `'cls'` | `'cls'` or `'mean'` |
| `transformer_use_action_history_token` | `false` | Action history as token vs post-concat |

### v4 — Extended Action Space

No new config fields. Uses `num_pad_multiples` (default 3: pad multiples 2,4,8) and `num_unroll_factors` (default 3: unroll factors 2,4,8) which are available in all configs.

---

## Directory Layout

```
results/my_experiment/
├── exec_times/                    # Shared across all implementations
│   ├── base.json                  # Unoptimized MLIR baseline (all benchmarks)
│   ├── base_train.json            # Train split
│   ├── base_eval.json             # Eval split
│   └── pytorch.json               # PyTorch baselines
├── old_agent/                     # Baseline RL
│   └── run_0/
│       ├── models/model_*.pt
│       ├── logs/train/ eval/
│       └── exec_data.json
├── v1_agent/                      # Hardware-aware v1
├── v2_agent/                      # Shaped-reward v2
├── v3_agent/                      # Transformer v3
├── v4_agent/                      # Extended-actions v4
└── comparison.csv                 # Multi-implementation comparison
```

---

## Training on a Single Benchmark (Quick Smoke Test)

To quickly test that a config works end-to-end, use a single benchmark:

```bash
# Create a tiny benchmark set
mkdir -p data/smoke
cp data/all/resnet18_sz224_bs1_conv_0.mlir data/smoke/

# Edit config to point at data/smoke and reduce iterations
export CONFIG_FILE_PATH=config/smoke.json
# ... set benchmarks_folder_path: "data/smoke", nb_iterations: 10
python scripts/get_base.py --config config/smoke.json
python scripts/split_json.py --config config/smoke.json
python scripts/train.py
```
