# Results Directory Architecture

## Directory Structure (Flat)

```
results/<experiment>/<agent_dir>/
├── train/                          # Training results
│   ├── results.json                # Cumulative {bench: {rewards, speedup, exec_time, cache_miss}}
│   └── checkpoint_100.json         # Snapshot every 100 iterations
├── eval/                           # Checkpoint evaluation results
│   ├── checkpoint_100.json         # {bench_name: exec_time_ns} per checkpoint
│   └── markers/<ckpt>/             # Eval crash-resilience markers
├── models/                         # Model checkpoints
│   ├── model_50.pt                 # Saved every 50 iterations
│   ├── model_100.pt
│   └── ...
└── logs/                           # Training & eval logs
    ├── exec_data.json              # Execution time cache
    ├── tags                        # Config tags
    ├── train/                      # entropy, reward, final_speedup
    ├── train_ppo/                  # policy_loss, value_loss, approx_kl, clip_frac
    └── eval/                       # eval_exec_times.json + per-benchmark files
```

## Key Changes from Old Structure

| Component | Old Location | New Location |
|-----------|-------------|-------------|
| Training markers | `global_markers/training/iter_N/<bench>` (one file per benchmark per iteration) | `train/checkpoint_N.json` (single cumulative JSON) |
| Live results | `global_markers/default/<bench>` | `train/results.json` (updated every iteration) |
| Eval markers | `global_markers/ckpt_N/<bench>` | `eval/markers/<bench>` (inside run dir) |
| Execution cache | `run_N/exec_data.json` | `run_N/logs/exec_data.json` |
| Model checkpoints | `model_0.pt` (index 0) | `model_1.pt` (index 1) |
| Impl subdir nesting | `<agent_dir>/<impl_subdir>/run_N/` | `<agent_dir>/run_N/` (removed nesting) |
| Eval post-process target | `<agent_dir>/eval/checkpoint_N.json` | `<agent_dir>/run_N/eval/checkpoint_N.json` |

## Train/Results JSON Format

`train/results.json` and `train/checkpoint_N.json` use the same format:
```json
{
  "bench_name_1": {
    "rewards": [0.0, -0.05, 0.02],
    "speedup": 2.5,
    "exec_time": 500000,
    "cache_miss": false
  },
  ...
}
```

`eval/checkpoint_N.json` stores benchmark → optimized execution time in nanoseconds:
```json
{
  "bench_name_1": 500000,
  "bench_name_2": null,
  ...
}
```

## Snapshot Logic

- Every training iteration updates `train/results.json` with the latest batch results
- Every 100 iterations, a copy is saved as `train/checkpoint_<N>.json`
- The final iteration always snapshots to `train/checkpoint_<last>.json`
- During eval, `eval/checkpoint_N.json` is saved after each checkpoint evaluation via `--checkpoint` post-processing

## Eval Crash Resilience

During checkpoint evaluation (`--checkpoint` mode), a temp `run_ckpt_N/` is created. After each benchmark executes:
1. Result is saved to `run_ckpt_N/eval/markers/<bench_name>`
2. On restart, benchmarks with existing markers are skipped
3. After all benchmarks complete, `eval_exec_times.json` is copied to `run_N/eval/checkpoint_N.json`

## FileLogger Singleton

`utils/file_logger.py` creates the run directory at `__init__`:
- `self.run_dir` = `<agent_dir>/run_N/` (auto-increments from existing runs)
- `self.models_dir` = `<run_dir>/models/`
- `self.logs_dir` = `<run_dir>/logs/`
- `self.train_dir` = `<run_dir>/train/`
- `self.eval_dir` = `<run_dir>/eval/`
- `self.train_results_file` = `<train_dir>/results.json` (initialized as `{}`)
- `self.exec_data_file` = `<logs_dir>/exec_data.json` (initialized as `{}`)

`FORCE_RUN_ID` env var controls run directory naming:
- `FORCE_RUN_ID=5` → `run_5/` (numeric, reuses existing)
- `FORCE_RUN_ID=ckpt_100` → `run_ckpt_100/` (non-numeric, creates temp dir for eval)

## Resuming Training

`train.sh --resume <path>` expects a `run_N/` path with `models/` subdirectory:
```bash
sbatch scripts/train/train.sh config/train/v4_7.json --resume results/new_dataset_results/v4_7_agent/run_0
```

Train loads the latest `model_*.pt` from `<path>/models/` and resumes from `step + 1`.

## Migration Script

`scripts/migrate_results.py` handled converting from old to new structure:
- Moved `impl_subdir/run_N/` → `<agent_dir>/run_N/`
- Converted `global_markers/training/iter_N/` → `train/checkpoint_N.json` (cumulative)
- Moved `models/` (agent root) → `run_0/models/`
- Moved `eval/checkpoint_*.json` → `run_0/eval/`
- Moved `eval/logs/` → `run_0/logs/eval/`
- Moved `exec_data.json` → `logs/exec_data.json`
