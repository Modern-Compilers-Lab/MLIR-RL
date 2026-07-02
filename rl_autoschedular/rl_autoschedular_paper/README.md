# MLIR-PAPER Agent (`rl_autoschedular_paper`)

Port of the published paper artifact (Tirichine et al.) into the MLIR-RL training framework.

**Source:** `/scratch/mb10856/MLIR-PAPER/mlir_rl_artifact/`
**Package:** `rl_autoschedular_paper/`
**Configs:** `config/paper_mohammed/`

## What Changed

All `mlir_rl_artifact` imports → `rl_autoschedular_paper`. Nothing else was modified.

## Structure

```
rl_autoschedular_paper/
├── model.py              # HiearchyModel (LSTM embedding + policy/value heads)
├── env.py                # RL environment (state transitions, reward)
├── ppo.py                # PPO training loop (collect, update, evaluate)
├── observation.py        # Observation tensor construction
├── state.py              # OperationState, BenchmarkFeatures, AST parsing
├── trajectory.py         # Trajectory storage, GAE, returns
├── transforms.py         # MLIR transform dialect passes (in-process Module)
├── execution.py          # MLIR compilation + execution (in-process)
├── benchmarks.py         # Benchmark loading + feature extraction
├── actions/              # 6 actions: NT, T, TP, TF, I, V
│   ├── base.py
│   ├── interchange.py    # 3 modes: enumerate, pointers, continuous
│   ├── tiling.py
│   ├── tiled_parallelization.py
│   ├── tiled_fusion.py
│   ├── vectorization.py
│   └── no_transformation.py
├── utils/                # Config, logging, Dask, process isolation
│   ├── config.py
│   ├── singleton.py
│   ├── file_logger.py
│   ├── dask_manager.py
│   ├── bindings_process.py
│   ├── gpu_occupier.py
│   └── log.py
├── train.py              # Training entry point
├── evaluate.py           # Eval all checkpoints
└── baseline.py           # Baseline timing
```

## Key Differences from MLIR-RL V0

| Aspect | Paper Agent (`rl_autoschedular_paper`) | V0 (`rl_autoschedular_v0`) |
|--------|---------------------------------------|---------------------------|
| **Code interface** | In-process MLIR `Module` objects | Code strings + subprocess isolation |
| **Execution** | `BindingsProcess.call()` | `multiprocessing.Process` + CLI fallback |
| **Transforms** | Mutate Module in-place | Shell out to subprocesses with temp files |
| **Interchange mode** | `"pointers"` (incremental permutation) | `"enumerate"` (all candidates) |
| **Max loop dims** | 12 loops, 12 load/store dims | 7 loops, 7 load/store dims |
| **Epsilon start** | 0.0 (no random exploration) | 0.5 (50% random) |
| **Experience replay** | None (replay_count=0) | replay_count=10 |
| **PPO batch size** | 64 | 32 |
| **Failed speedup** | Returns 1.0 | Returns 0.0 |
| **Speedup reward** | No clamping | Clamps exec times to ≥1 |
| **Multi-run eval** | Single run | Configurable runs with aggregation |
| **Eval reporting** | Arithmetic mean | Geometric + arithmetic mean |
| **Model checkpoints** | Every 5 iterations | Every 50 iterations |

## Key Differences from V4.5+ (Transformer)

| Aspect | Paper Agent | V4.5+ |
|--------|-------------|-------|
| **Encoder** | LSTM (512→411) | Transformer (d=256, 8 heads, 3 layers) |
| **HW features** | None | 7-dim (L1/L2/L3, cores, SIMD, clock) |
| **Reward** | Sparse terminal only | Shaped + terminal (or sparse in v4.9) |
| **Observation** | ~2100 dims (flat tensor) | Transformer tokenized input |

## Training

```bash
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
export CONFIG_FILE_PATH=config/paper_mohammed/paper.json

python -m rl_autoschedular_paper.train
```

## Evaluation

```bash
export CONFIG_FILE_PATH=config/paper_mohammed/paper_eval.json

python -m rl_autoschedular_paper.evaluate
```

## Slurm

```bash
# Train
sbatch scripts/train/train.sh config/paper_mohammed/paper.json

# Eval
sbatch --cpus-per-task=12 --mem=16G --time=04:00:00 \
  scripts/eval/eval.sh config/paper_mohammed/paper_eval.json
```

## Prerequisites

The paper agent uses **in-process MLIR Python bindings** (not subprocess isolation). If bindings are broken symlinks from another user's LLVM build, follow the LLVM Build Gotchas in `AGENTS.md` to fix them.
