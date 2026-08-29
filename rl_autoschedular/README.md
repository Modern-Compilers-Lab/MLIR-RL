# `rl_autoschedular/` — The RL Autoscheduler (Phase 1)

**Phase 1 of the [MLIR-RL](../README.md) research programme, and the system described in the
published paper.**

A deep reinforcement learning agent that optimizes MLIR loop nests by choosing a *schedule*: an
ordered sequence of Transform-dialect transformations with parameters. The action space is
**hand-designed** (seven actions), and the policy is a **custom hierarchical PPO** implementation —
an action head that picks the transformation, plus per-action parameter heads.

> ## Reproducing the paper
>
> **Use the official artifact, not this directory:**
> **<https://github.com/mohph197/MLIR-RL-artifact>**
>
> It ships a Dockerfile, pinned dependencies (Python 3.11, LLVM/MLIR built from source, Clang
> 21.1.5), pre-trained model checkpoints, and three scripts: `scripts/train.sh`,
> `scripts/evaluate.sh`, and `scripts/paper.sh` — the last regenerates the paper's speedup numbers
> and figures into `paper/results/` and `paper/figures/`. It evaluates on the training set, the
> evaluation set, full models, neural-network operators, and Lattice QCD workloads.
>
> **This directory is the research working copy.** It has diverged from the artifact and shares code
> with the offline-RL work in [../iql/](../iql/). Use it to *understand or extend* the system; use
> the artifact to *reproduce results*.

---

## Table of contents

1. [Architecture](#1-architecture)
2. [The hand-designed action space](#2-the-hand-designed-action-space)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Running](#5-running)
6. [Known-stale entry points](#6-known-stale-entry-points)

---

## 1. Architecture

Unlike phase 2, this system is **not** built on Gymnasium or Stable-Baselines3 — the environment,
the model, and PPO are all implemented here directly.

```
  Benchmarks ──► Env.reset ──► OperationState ──► Observation ──► HiearchyModel
                                    ▲                                  │
                                    │                              action + params
                                    │                                  │
                              Env.step  ◄──────────── ActionSpace ◄─────┘
                                    │
                              Execution (MLIR bindings) ──► time ──► reward
                                    │
                              TrajectoryData ──► ppo_update / value_update
```

| File | Role |
|---|---|
| [benchmarks.py](benchmarks.py) | `Benchmarks` — loads kernels and their pre-measured execution times; train / eval splits |
| [state.py](state.py) | `OperationState`, `BenchmarkFeatures`, `OperationFeatures`, `NestedLoopFeatures`, `OperationType`, `IteratorType`; `extract_bench_features_from_code` parses MLIR into features |
| [observation.py](observation.py) | `Observation` assembled from parts: `OpFeatures`, `ProducerOpFeatures`, `ActionHistory`, `ActionMask`, `NumLoops`. `Observation.get_parts(obs, *parts)` slices it |
| [env.py](env.py) | `Env.reset(benchs, bench_idx)` → `OperationState`; `Env.step(state, action)` → next state. Also `get_next_op_state` for multi-operation benchmarks. Episodes end on a terminal action, on failure, or at `cfg.truncate` steps |
| [actions/](actions/) | The action space — see below |
| [model.py](model.py) | `HiearchyModel` (the hierarchical actor-critic used by `train.py`), plus `ValueModel`, `PolicyModel`, `LSTMEmbedding` |
| [ppo.py](ppo.py) | `collect_trajectory`, `ppo_update`, `value_update`, `evaluate_benchmarks` |
| [trajectory.py](trajectory.py) | `TrajectoryData` — rollout storage, concatenation, copying (used by `reuse_experience`) |
| [execution.py](execution.py) | Compiles and runs MLIR through the Python bindings' `ExecutionEngine`, linking `MLIR_SHARED_LIBS` |
| [transforms.py](transforms.py) | Transform-dialect emission helpers |

Cross-cutting code lives at the repository root:

- [../utils/config.py](../utils/config.py) — the `Config` singleton, loaded from the JSON file named
  by `CONFIG_FILE_PATH` (resolved in [../utils/keys.py](../utils/keys.py)).
- [../utils/file_logger.py](../utils/file_logger.py) — per-run directory, model dir, execution-data cache.
- [../utils/dask_manager.py](../utils/dask_manager.py) — distributes environment steps across Dask
  workers; reads `DASK_NODES`.
- [../utils/data_collector.py](../utils/data_collector.py) — `OfflineDataset`, used by [../iql/](../iql/).
- Neptune experiment tracking ([../neptune_sync.py](../neptune_sync.py)).

## 2. The hand-designed action space

Seven actions, listed in `ActionSpace.supported_actions` ([actions/__init__.py](actions/__init__.py)).
Each has a one- to three-letter **symbol** used by the `order` field of the config file:

| Symbol | Action | File |
|---|---|---|
| `!` | `NoTransformation` — terminal no-op | [actions/no_transformation.py](actions/no_transformation.py) |
| `T` | `Tiling` | [actions/tiling.py](actions/tiling.py) |
| `TP` | `TiledParallelization` | [actions/tiled_parallelization.py](actions/tiled_parallelization.py) |
| `TPF` | `TiledFusion` — tiles and fuses with a producer op | [actions/tiled_fusion.py](actions/tiled_fusion.py) |
| `I` | `Interchange` | [actions/interchange.py](actions/interchange.py) |
| `V` | `Vectorization` | [actions/vectorization.py](actions/vectorization.py) |
| `I2C` | `Img2Col` — im2col lowering for convolutions | [actions/img2col.py](actions/img2col.py) |

Each action subclasses `Action` ([actions/base.py](actions/base.py)) and declares `params_size()`,
`mask_size()`, `history_size()`, so `ActionSpace` can lay out one flat parameter vector and one flat
mask vector across all actions (`cumulative_params_sizes`, `cumulative_mask_sizes`,
`cumulative_history_sizes`).

Two actions are special-cased in the masking logic:

- `Interchange` can be **incomplete** — when `interchange_mode` is `pointers`, a permutation is built
  over several steps, and the mask forces `Interchange` again until it is complete.
- `Img2Col` and `TiledFusion` are gated on the operation type and on having a producer.

**This hand-written action set is exactly what phase 2 ([../llm_action/](../llm_action/)) automates.**

## 3. Installation

### Prerequisites

This component needs **LLVM/MLIR built from source** (route B in the
[root README](../README.md#route-b--llvmmlir-built-from-source)), because the C++ tools below are
`add_llvm_executable` targets. Build LLVM first, then:

```bash
conda create -n llvm-build python=3.11
conda activate llvm-build
pip install -r ../requirements.txt
```

### Building the C++ tools

Three standalone MLIR CMake projects live in [../tools/](../tools/):

| Tool | Binary | Purpose |
|---|---|---|
| [../tools/ast_dumper/](../tools/ast_dumper/) | `AstDumper` | Dumps loop bounds, iterator types and memory-access affine maps from an MLIR file — this is how the observation is built |
| [../tools/vectorizer/](../tools/vectorizer/) | `Vectorizer` | Optional C++ path for the vectorization action (`use_vectorizer`) |
| [../tools/pre_vec/](../tools/pre_vec/) | `PreVec` | Pre-vectorization analysis |

Build each the same way:

```bash
export LLVM_BUILD_PATH=/path/to/llvm-project/build

for tool in ast_dumper vectorizer pre_vec; do
  cmake -S tools/$tool -B tools/$tool/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR="$LLVM_BUILD_PATH/lib/cmake/mlir" \
    -DLLVM_DIR="$LLVM_BUILD_PATH/lib/cmake/llvm"
  cmake --build tools/$tool/build
done
```

Binaries appear at `tools/<tool>/build/bin/{AstDumper,Vectorizer,PreVec}`. `tools/*/build` is
gitignored.

> `AstDumper` is also required by [../llm_action/](../llm_action/) for its observation encoder — it
> is the one piece of phase-1 tooling phase 2 still depends on.

### Environment file

Copy [../.env.example](../.env.example) to `../.env` and fill it in. Run from the **repository root**.

| Variable | Meaning |
|---|---|
| `CONDA_ENV` | Path to (or name of) the conda environment |
| `LLVM_BUILD_PATH` | Your `llvm-project/build` directory |
| `MLIR_SHARED_LIBS` | Comma-separated `libomp.so,libmlir_c_runner_utils.so,libmlir_runner_utils.so` under `$LLVM_BUILD_PATH/lib` — read by [execution.py](execution.py) |
| `AST_DUMPER_BIN_PATH` | `tools/ast_dumper/build/bin/AstDumper` |
| `VECTORIZER_BIN_PATH` | `tools/vectorizer/build/bin/Vectorizer` |
| `PRE_VEC_BIN_PATH` | `tools/pre_vec/build/bin/PreVec` |
| `CONFIG_FILE_PATH` | Absolute path to the JSON config (see below) |
| `NEPTUNE_PROJECT`, `NEPTUNE_TOKEN` | Experiment tracking |

Also export the MLIR toolchain onto your paths:

```bash
export PATH="$LLVM_BUILD_PATH/bin:$PATH"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PYTHONPATH"
```

`.env` is gitignored — never commit it.

## 4. Configuration

Everything is driven by one JSON file, pointed at by `CONFIG_FILE_PATH`. Examples:
[../config/config.json](../config/config.json) and [../config/example.json](../config/example.json).
Keys are loaded into the `Config` singleton in [../utils/config.py](../utils/config.py); unknown keys
are ignored and missing keys fall back to the defaults defined there.

### Observation and action shape

| Key | Example | Meaning |
|---|---|---|
| `max_num_loops` | 7 | Maximum nesting depth the observation can encode |
| `max_num_stores_loads` | 7 | Maximum number of load/store accesses encoded |
| `max_num_load_store_dim` | 7 | Maximum rank of an indexed buffer |
| `num_tile_sizes` | 7 | Size of the tile-size vocabulary per loop |
| `vect_size_limit` | 512 | Reject vectorizations above this many elements |
| `normalize_bounds` | `"max"` | Loop-bound normalization: `none` / `max` / `log` |

### Action schedule

| Key | Example | Meaning |
|---|---|---|
| `order` | `[["!","TPF"], ["!","I2C","TPF"], …]` | Per-step action mask. See below |
| `interchange_mode` | `"pointers"` | `enumerate` (all permutations as classes) / `pointers` (built over several steps) / `continuous` |
| `use_img2col` | `true` | Enable the `I2C` action |
| `truncate` | 5 | Maximum schedule length (steps per operation) |

**`order` semantics** ([actions/__init__.py](actions/__init__.py)) — one entry per step, using the
action symbols from [§2](#2-the-hand-designed-action-space):

- `[]` (empty) — allow every action at this step.
- `["!", "A", "B"]` — leading `"!"` makes it a **denylist**: allow everything *except* `A` and `B`.
- `["A", "B"]` — an **allowlist**: only `A` and `B`.
- The list must be at least `truncate` long and must end with a step that permits a terminal action,
  or the environment raises `"actions order must be ended with a terminal action"`.

So `[["!","TPF"], ["!","I2C","TPF"], …]` means: at step 0 anything but tiled fusion; at every
subsequent step anything but im2col and tiled fusion.

### PPO and training

| Key | Example | Meaning |
|---|---|---|
| `nb_iterations` | 10000 | Outer training loop iterations |
| `bench_count` | 64 | Benchmarks sampled per trajectory |
| `ppo_epochs` / `ppo_batch_size` | 4 / 32 | PPO update epochs and minibatch size |
| `value_epochs` / `value_batch_size` / `value_coef` / `value_clip` | 0 / 32 / 0.5 / false | Separate value-fitting pass; `value_epochs: 0` disables it |
| `entropy_coef` | 0.01 | Exploration bonus |
| `lr` | 0.001 | Adam learning rate |
| `normalize_adv` | `"standard"` | Advantage normalization |
| `exploration` | `["entropy"]` | `entropy` and/or `epsilon`; `init_epsilon` sets the start value |
| `reuse_experience` | `"none"` | Concatenate the previous trajectory into the current one |
| `replay_count` | 10 | Replay buffer size when experience reuse is on |

### Data and logging

| Key | Meaning |
|---|---|
| `json_file` / `eval_json_file` | Baseline execution times for the train / eval splits |
| `benchmarks_folder_path` | Kernel directory (may be empty when times come from the JSON files) |
| `split_ops` | Treat each operation in a benchmark as its own episode |
| `results_dir` | Where run directories are created |
| `save_model_every` / `evaluate_every` | Checkpoint and evaluation cadence (in iterations) |
| `tags` | Neptune experiment tags |
| `debug` | Enables `torch.autograd.set_detect_anomaly` and extra logging |
| `main_exec_data_file` | Optional shared execution-time cache across runs |

## 5. Running

All commands run from the **repository root**.

### Train

```bash
sbatch scripts/train_example.sh
```

[../scripts/train_example.sh](../scripts/train_example.sh) requests 28 exclusive cores, activates
`$CONDA_ENV_NAME`, sets `OMP_NUM_THREADS=12` and `CONFIG_FILE_PATH`, then runs
[../train.py](../train.py). Interactively:

```bash
export CONFIG_FILE_PATH=config/config.json
export DASK_NODES=1          # required by utils/dask_manager.py
python train.py
```

`train.py` loads train and eval `Benchmarks` onto the Dask workers, builds `HiearchyModel`, and loops
`nb_iterations` times over: collect trajectory → optional value update → PPO update → periodic
checkpoint and evaluation. Checkpoints are written as `model_<step>.pt` into the run's `models/`
directory; a run summary is printed with elapsed time and ETA per iteration.

Debug logs go to `logs/<SLURM_JOB_NAME>_<SLURM_JOB_ID>.debug`. `SLURM_JOB_ID` must be set — when
running interactively, export a dummy value.

### Evaluate

```bash
export CONFIG_FILE_PATH=config/config.json
export EVAL_DIR=results/<run>/models      # a directory of model_<step>.pt files
python evaluate.py
```

[../evaluate.py](../evaluate.py) sorts the checkpoints by step and evaluates every one in sequence
against the eval split, so you get the whole learning curve in real measured speedups.

### Measure baselines

```bash
python get_base.py <folder-of-mlir-files>
```

[../get_base.py](../get_base.py) executes every `.mlir` in the folder and writes
`execution_times_train.json` / `execution_times_eval.json` one level up, splitting according to
`../benchmarks_split.json`. Failed executions are recorded as `-1`. It writes after every file, so it
is safe to interrupt and resume.

### Other utilities

| Script | Purpose |
|---|---|
| [../gen.py](../gen.py) | Generates benchmarks and dumps `OperationFeatures` / `NestedLoopFeatures` into `data/features` |
| [../fill_db.py](../fill_db.py) | Populates a trajectory database by rolling out random/uniform action sequences (used to seed the offline dataset for [../iql/](../iql/)) |
| [../scripts/neptune-sync.sh](../scripts/neptune-sync.sh) | Uploads offline Neptune runs from `.neptune/` |
| [../scripts/setup_env.sh](../scripts/setup_env.sh) | Exports `PYTHONPATH` and `MLIR_SHARED_LIBS` for a conda-provided MLIR |

## 6. Known-stale entry points

The working copy has drifted. **The entry points that work are [../train.py](../train.py) and
[../evaluate.py](../evaluate.py)** — they construct `utils.config.Config()` and
`utils.file_logger.FileLogger()` directly.

These do **not** import cleanly on this branch:

| File | Problem |
|---|---|
| [../train_ppo.py](../train_ppo.py) | `from rl_autoschedular import config as cfg, file_logger as fl` — [`__init__.py`](__init__.py) only exports `device` |
| [../fill_db.py](../fill_db.py) | same stale import; also reads `cfg.num_transformations`, which is not in the current `Config` |
| [../demo.py](../demo.py), [../demo.ipynb](../demo.ipynb) | call `Env(is_training=False)` and `evaluate_benchmark`, neither of which matches the current `Env` / `ppo.py` signatures (`Env.reset(benchs, bench_idx)`, `evaluate_benchmarks(model, data)`) |
| [../train_iql_offline.py](../train_iql_offline.py), [../train_iql_online.py](../train_iql_online.py) | same stale import, plus the issues noted in [../iql/README.md](../iql/README.md) |

These are leftovers from a branch merge, not deep breakage — fixing them is mostly re-pointing
imports at `utils.config` / `utils.file_logger` and updating a few call signatures. Flagged here so
nobody loses a day to it.
