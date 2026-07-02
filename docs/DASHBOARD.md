# MLIR-RL Evaluation Dashboard

## 1. Overview

This Streamlit dashboard visualizes the evaluation results of the MLIR-RL system — a deep Reinforcement Learning environment for automatic loop-level code optimization in the MLIR compiler. The agent optimizes MLIR Linalg code by selecting sequences of transformations (tiling, parallelization, fusion, interchange, vectorization) using a PPO-based actor-critic policy.

The dashboard is designed to answer three core questions:

- **How do RL-optimized execution times compare to PyTorch baselines?**
- **How much speedup does the RL agent achieve over the unoptimized MLIR baseline?**
- **How does the agent's performance evolve over training?**

---

## 2. Results Folder Structure

The dashboard reads from the canonical directory layout under `results/`:

Example:

```
results/
└── <experiment>/                        # e.g. albert, mehdi
    ├── exec_times/
    │   ├── old_base.json
    │   ├── old_base_train.json
    │   ├── old_base_eval.json
    │   ├── new_base.json
    │   ├── new_base_train.json
    │   ├── new_base_eval.json
    │   └── pytorch.json               # PyTorch reference times (eager, compile, jit)
    ├── old_agent/
    │   ├── run_0/
    │   │   ├── logs/
    │   │   │   ├── eval/
    │   │   │   │   ├── exec_time/     # one file per benchmark, m floats = exec time per eval step
    │   │   │   │   ├── speedup/       # one file per benchmark, m floats = speedup per eval step
    │   │   │   │   ├── average_speedup
    │   │   │   │   ├── final_speedup
    │   │   │   │   ├── cumulative_reward
    │   │   │   │   ├── reward
    │   │   │   │   └── entropy
    │   │   │   ├── train/
    │   │   │   └── train_ppo/
    │   │   └── models/
    │   └── run_1/
    │       └── ...
    └── new_agent/
        └── run_N/
            └── ...
```

### Key Data Formats

**`exec_times/old_base_eval.json`** and **`exec_times/new_base_eval.json`** — flat JSON files, keys are benchmark identifiers and values are integer exec times in µs. These are unoptimized MLIR baselines (O3, no loop-level transforms) for old and new implementations.

```json
{
  "albert_sl128_bs1_generic_0": 18265,
  "albert_sl128_bs16_batch_matmul_1": 186205951
}
```

Benchmark key format: `{model}_{sl{seq_len}}_{bs{batch_size}}_{op_type}_{index}`. The `sl{seq_len}` component is optional (NLP models only).

**`exec_times/pytorch.json`** — same keys, values are objects with three PyTorch execution modes in µs.

```json
{
  "albert_sl128_bs16_batch_matmul_0": {
    "eager": 59087,
    "compile": 156918,
    "jit": 65007
  }
}
```

**`logs/eval/exec_time/<benchmark>`** — plain text, one float per line. Each line is the RL-optimized execution time at a given eval step. The minimum across all lines is used as the agent's best result.

**`logs/eval/speedup/<benchmark>`** — plain text, one float per line. Each line is the speedup ratio (`baseline / optimized`) at a given eval step.

**Plain-text log files** — one float per line, indexed as step 0, 1, 2, …: `average_speedup`, `final_speedup`, `reward`, `cumulative_reward`, `entropy`.

---

## 3. Benchmark Categories

Benchmarks are automatically classified into three categories based on the paper's evaluation setup:

| Category | Description | Examples |
|----------|-------------|---------|
| **DL Operator** | Single deep learning operators | `matmul`, `conv_2d_nchw_fchw`, `generic (elementwise)` |
| **DL Model** | Full neural network models | `albert`, `resnet`, `mobilenet`, `gpt2` |
| **LQCD** | Lattice QCD applications | `dibaryon`, `hexaquark` |

---

## 4. Model Family Normalization

Benchmark keys are parsed to extract a `model_family` and a `sub_family`. Model families are normalized to collapse size/variant suffixes into canonical names:

| Raw key prefix | Normalized family | Sub-family examples |
|----------------|-------------------|---------------------|
| `resnet18_sz224`, `resnet50_sz160` | `resnet` | `resnet18`, `resnet50` |
| `mobilenet_v2_sz160`, `mobilenet_v3_sz192` | `mobilenet` | `mobilenet_v2`, `mobilenet_v3` |
| `convnext_tiny_sz224` | `convnext` | `convnext_tiny` |
| `efficientnet_b0_sz224` | `efficientnet` | `efficientnet_b0` |
| `densenet121_sz224` | `densenet` | `densenet121` |
| `lstm_seq2seq` | `lstm_seq2seq` | *(own family, not merged with lstm)* |
| `vit_b_16`, `vit_l_32` | `vit` | `vit_b`, `vit_l` |
| `albert`, `bert`, `gpt2` | unchanged | — |

---

## 5. Dashboard Layout

### Sidebar

The sidebar controls what data is loaded and how it is filtered globally across all tabs.

- **Experiment** — selects the top-level results folder (e.g. `albert`, `mehdi`)
- **Run** — selects the run id (`run_0`, `run_1`, …); the dashboard loads matching runs from `old_agent/` and `new_agent/` when available
- **Benchmark Category** — multiselect: `DL Operator`, `DL Model`, `LQCD`
- **Batch Size** — multiselect, parsed from benchmark keys
- **Op Type** — multiselect, parsed from benchmark keys
- **Min RL Speedup vs MLIR Baseline** — slider to filter out low-performing benchmarks
- **Reload Data** — clears `@st.cache_data` to pick up new Slurm job results without restarting

### Summary Metrics (top of page)

Four metric cards shown above the tabs, computed over the currently filtered dataset:

| Metric | Definition |
|--------|-----------|
| Avg Old RL vs PyTorch Eager | Mean of `pytorch_eager / rl_old_optimized` |
| Avg New RL vs PyTorch Eager | Mean of `pytorch_eager / rl_new_optimized` |
| Old RL Beats All PyTorch Modes | Count where old RL is faster than eager, compile, and jit |
| New RL Beats All PyTorch Modes | Count where new RL is faster than eager, compile, and jit |

> **> 1 means RL is faster than PyTorch. < 1 means PyTorch is faster than RL.**

---

## 6. Tabs

### Tab 1 — Execution Times

Compares four execution time sources per model family, all in µs:

- **Old RL Optimized** — `min` across eval steps from `old_agent/run_i/logs/eval/exec_time/`
- **New RL Optimized** — `min` across eval steps from `new_agent/run_i/logs/eval/exec_time/`
- **MLIR Baseline** — old/new baseline eval files from `exec_times/`
- **PyTorch Eager / Compile / JIT** — from `pytorch.json`

The tab contains two charts. The first shows average exec time per model family with a multiselect to filter families. The second is a sub-family drill-down: select a family (e.g. `resnet`) to see individual sub-families (`resnet18`, `resnet50`) compared side by side.

The data table includes old/new-specific fields such as `rl_old_speedup_over_baseline`, `rl_new_speedup_over_baseline`, `rl_old_vs_eager`, and `rl_new_vs_eager`.

### Tab 2 — Speedup Analysis

Focuses on the RL agent's speedup over the MLIR baseline, independent of PyTorch.

An **Operation Type selectbox** lets the user pick a single op type (e.g. `matmul`, `conv_2d_nchw_fchw`). The chart then shows the **average best speedup per model family** for that op type — answering "which model families benefit most from RL optimization for this operation?".

Below the chart, a line chart shows how `average_speedup` evolves over eval steps across the whole run, followed by a per-benchmark overlay where the user can select individual benchmarks to compare their speedup trajectories.

### Tab 3 — Training Curves

Line charts for each RL training signal, one per chart:

- **Reward** — per-step reward from `logs/eval/reward`
- **Cumulative Reward** — from `logs/eval/cumulative_reward`
- **Policy Entropy** — from `logs/eval/entropy`
- **Train Reward / Entropy** — from `logs/train/` if available

All x-axes represent training step index. A JSON export of all curves is available.

### Tab 4 — Checkpoints

Reads `.pt` checkpoint files from `old_agent/run_i/models/` and `new_agent/run_i/models/`, and pairs them with per-benchmark exec times from each implementation's `logs/eval/exec_time/`.

Shows two charts: average exec time across all benchmarks over checkpoints, and a per-benchmark overlay where the user selects which benchmarks to compare. Hovering shows the actual checkpoint filename. An expandable section lists all checkpoint files, and a CSV export is available.

---

## 7. Speedup Semantics

All speedups in this dashboard are ratios over the **unoptimized MLIR baseline** (`old_base_eval.json` or `new_base_eval.json`), which is MLIR code compiled with O3 but without any loop-level transformations (no tiling, no parallelization, no fusion, no interchange, no vectorization).

| Column | Formula | Interpretation |
|--------|---------|----------------|
| `rl_old_speedup_over_baseline` | `old_baseline / rl_old_optimized` | Old RL improvement over old baseline |
| `rl_new_speedup_over_baseline` | `new_baseline / rl_new_optimized` | New RL improvement over new baseline |
| `rl_old_vs_eager` | `pytorch_eager / rl_old_optimized` | > 1: Old RL faster than PyTorch Eager |
| `rl_old_vs_compile` | `pytorch_compile / rl_old_optimized` | > 1: Old RL faster than PyTorch Compile |
| `rl_old_vs_jit` | `pytorch_jit / rl_old_optimized` | > 1: Old RL faster than PyTorch JIT |
| `rl_new_vs_eager` | `pytorch_eager / rl_new_optimized` | > 1: New RL faster than PyTorch Eager |
| `rl_new_vs_compile` | `pytorch_compile / rl_new_optimized` | > 1: New RL faster than PyTorch Compile |
| `rl_new_vs_jit` | `pytorch_jit / rl_new_optimized` | > 1: New RL faster than PyTorch JIT |

`rl_old_optimized_us` and `rl_new_optimized_us` are the **minimum exec times across eval steps** for each implementation.

---

## 8. Running the App

```bash
cd MLIR-RL/dashboard
pip install -r requirements.txt --break-system-packages
streamlit run dashboard.py --server.fileWatcherType none
```

The `--server.fileWatcherType none` flag is required on Slurm clusters where the inotify watch limit is typically exhausted. The **Reload Data** button in the sidebar handles refreshing results when new jobs finish.

### Streamlit Config (recommended)

Create `~/.streamlit/config.toml`:

```toml
[server]
fileWatcherType = "none"

[theme]
base = "light"
primaryColor = "#1a7f4b"
```

---

## 9. File Structure

```
MLIR-RL/
├── results/                   # generated by Slurm training jobs
│   └── <experiment>/
│       ├── exec_times/
│       ├── old_agent/
│       └── new_agent/
└── dashboard/
    ├── dashboard.py           # Streamlit app (single file)
    ├── dashboard.md           # this file
    └── requirements.txt       # streamlit>=1.32, pandas>=2.0, plotly>=5.18
```

---

## 10. Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
plotly>=5.18.0
```