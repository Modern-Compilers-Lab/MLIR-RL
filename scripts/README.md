# scripts/ — MLIR-RL Orchestration

No files at root. Everything organized into subfolders by purpose.

## Directory Layout

```
scripts/
├── train/           Training entry points + Slurm wrappers
├── eval/            Evaluation + ablation eval entry points + Slurm wrappers
├── baseline/        MLIR & PyTorch baseline timing scripts
├── full_model/      Full-model end-to-end optimization
├── checkpoint/      Checkpoint scan + results merger
├── data/            Dataset processing (split, analysis)
└── utils/           Sanity checks + workflow helpers
```

## Subfolder Details

### `train/`
| File | Purpose |
|------|---------|
| `train.py` | PPO training loop (core entry point) |
| `train.sh` | Slurm training wrapper (also handles array jobs) |

### `eval/`
| File | Purpose |
|------|---------|
| `eval.py` | Evaluation loop (core entry point) |
| `ablation_eval.py` | Ablation study evaluation (requires `EVAL_DIR`) |
| `eval.sh` | Generic Slurm eval (resolves impl from config) |
| `run_rl_eval.sh` | RL eval via `optimize_model_via_blocks.py` |
| `submit_checkpoint_evals.sh` | Batch submit checkpoint evaluations |

### `baseline/`
| File | Purpose |
|------|---------|
| `get_base.py` | MLIR baseline execution timing |
| `get_pytorch_times.py` | Block-level PyTorch timing (eager + JIT per op) |
| `get_pytorch_baselines.py` | **Canonical** PyTorch eager + JIT timing for 22 models |
| `get_base.sh` | Slurm wrapper for `get_base.py` (chunked) |
| `get_pytorch_times.sh` | Slurm wrapper for `get_pytorch_times.py` (chunked) |

### `full_model/`
| File | Purpose |
|------|---------|
| `optimize_full_model.py` | Main orchestrator for full-model RL optimization |
| `optimize_model_via_blocks.py` | Block-based fallback optimization |
| `preprocess_model.py` | Runs C++ AST dumper to tag linalg ops |
| `add_timing_wrapper.py` | Wraps @main with @nanoTime() |
| `optimize_full_model.sh` | Slurm array job wrapper (tasks 0–19) |
| `merge_full_model_results.sh` | Merges chunk files into unified JSON + CSV |
| `get_pytorch_full_times.sh` | Full PyTorch baseline timing |

### `checkpoint/`
| File | Purpose |
|------|---------|
| `merge_ckpt_scan.py` | Merges per-model checkpoint scan results |
| `ckpt_scan_all.sh` | Full checkpoint scan (all models, all ckpts + merge) |
| `submit_ckpt_scan.sh` | Submit checkpoint scan as Slurm array job |

### `data/`
| File | Purpose |
|------|---------|
| `split_json.py` | Stratified train/eval split of benchmark JSON |
| `build_checkpoint_comparison.py` | Merges full-model + block results into comparison CSV/JSON |

### `utils/`
| File | Purpose |
|------|---------|
| `test_torch_mlir_compile.py` | MLIR binding sanity check |
| `test_torch_mlir.sh` | Shell wrapper for the above |
| `submit_and_monitor.sh` | Submit Slurm job + auto-tail output |


## Standard Workflow

### Old Dataset

```bash
# 1. MLIR baseline
sbatch scripts/baseline/get_base.sh config/old_dataset/train/v4_5.json

# 2. PyTorch baseline
sbatch scripts/baseline/get_pytorch_times.sh config/old_dataset/train/v4_5.json

# 3. Split train/eval
python scripts/data/split_json.py config/old_dataset/train/v4_5.json

# 4. Train
sbatch scripts/train/train.sh config/old_dataset/train/v4_5.json

# 5. Evaluate
sbatch scripts/eval/eval.sh config/old_dataset/train/v4_5.json
```

### New Dataset

```bash
# 1. MLIR baselines (3 directories)
python scripts/baseline/get_base.py --config config/new_dataset/train/v4_5.json
# Repeat for eval and eval_full

# 2. Train V4.5 on Bergamo
sbatch scripts/train/train.sh config/new_dataset/train/v4_5.json

# 3. Eval V4.5 on Bergamo (5-run median)
sbatch scripts/eval/eval.sh config/new_dataset/eval/v4_5_eval.json

# 4. Full model eval (all ops)
sbatch scripts/eval/eval.sh config/new_dataset/full_model/v4_5_eval_full.json
```
