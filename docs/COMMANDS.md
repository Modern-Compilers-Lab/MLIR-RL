# MLIR-RL Project Commands Reference

This document describes all commands used in the MLIR-RL project, including training, evaluation, data generation, baseline measurement, reporting, and plotting.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Training](#training)
3. [Evaluation](#evaluation)
4. [Data Generation](#data-generation)
5. [Baseline Measurement](#baseline-measurement)
6. [Reporting & Analysis](#reporting--analysis)
7. [Plotting & Visualization](#plotting--visualization)
8. [Full Model Optimization](#full-model-optimization)
9. [Utilities](#utilities)

---

## Prerequisites

### Environment Setup (Interactive Use Only)

```bash
source ~/envs/mlir/bin/activate
set -a && source .env && set +a
export CONFIG_FILE_PATH=config/new_dataset/train/v4_7.json
```

**Note:** Slurm scripts (`train.sh`, `eval.sh`) handle `.env` and conda activation internally. You only need the steps above for running Python directly.

---

## Training

### Core Training Commands

#### Launch Training (Slurm)

```bash
# Basic training with config
sbatch scripts/train/train.sh config/new_dataset/train/v4_7.json

# Train specific implementation
sbatch scripts/train/train.sh config/new_dataset/train/v4_7.json rl_autoschedular_v4_5

# Resume training from checkpoint
sbatch scripts/train/train.sh config/new_dataset/train/v4_7.json \
  --resume results/new_dataset_results/v4_7_agent

# Force fresh training (overwrite existing results)
FORCE_NEW=1 sbatch scripts/train/train.sh config/new_dataset/train/v4_7.json
```

#### Training on Condo Partition (48h limit)

```bash
sbatch scripts/train/train_condo.sh config/new_dataset/train/v4_7.json
```

#### Direct Python Training (Debugging)

```bash
python scripts/train/train.py
```

**Environment Variables:**
- `CONFIG_FILE_PATH`: Path to config JSON (required)
- `AUTOSCHEDULER_IMPL`: Implementation package (optional, overrides config)
- `RESUME_FROM`: Resume directory path (optional)
- `FORCE_NEW`: Set to `1` to overwrite existing results

### Training Config Structure

```json
{
  "benchmarks_folder_path": "data/new_dataset/all/train",
  "bench_count": 64,
  "nb_iterations": 10000,
  "ppo_epochs": 4,
  "ppo_batch_size": 64,
  "lr": 0.001,
  "truncate": 5,
  "results_dir": "results/new_dataset_results/v4_7_agent",
  "implementation": "rl_autoschedular_v4_5",
  "json_file": "results/new_dataset_results/baselines/mlir/train_base.json",
  "hardware_auto_detect": true,
  "reward_shaping_scale": 0.05,
  "reward_shaping_clip": 0.1,
  "transformer_dim": 256,
  "transformer_heads": 8,
  "transformer_layers": 3,
  "ppo_clip_range": 0.2,
  "gae_lambda": 0.95,
  "max_grad_norm": 0.5
}
```

---

## Evaluation

### Core Evaluation Commands

#### Evaluate All Checkpoints

```bash
sbatch scripts/eval/eval.sh config/new_dataset/eval/v4_7_eval.json
```

#### Evaluate Single Checkpoint

```bash
sbatch scripts/eval/eval.sh config/new_dataset/eval/v4_7_eval.json --checkpoint 500
```

#### Evaluate with Custom Model Directory

```bash
EVAL_DIR=/path/to/models sbatch scripts/eval/eval.sh config.json
```

#### Ablation Evaluation (V0, V4.5 variants)

```bash
sbatch scripts/eval/ablation_eval.sh config/ablation/v45_no_hw.json
```

### Cluster-Specific Evaluation

```bash
# Bergamo (300GB, 128 CPUs)
sbatch scripts/eval/bergamo_eval.sh config.json

# Dalma (Intel Xeon)
sbatch scripts/eval/dalma_eval.sh config.json

# Jubail (AMD EPYC)
sbatch scripts/eval/jubail_eval.sh config.json
```

### Submit Multiple Checkpoint Evaluations

```bash
# Submit 15 checkpoint eval jobs (v4_6, v4_7, v4_8 x checkpoints 100-500)
bash scripts/eval/submit_checkpoint_evals.sh
```

### Evaluation Environment Variables

- `EVAL_DIR`: Override model directory (default: from config)
- `EVAL_START`: Start checkpoint number
- `EVAL_END`: End checkpoint number
- `EVAL_STRIDE`: Checkpoint stride (default: 100)
- `EVAL_LAST_ONLY`: Evaluate only last checkpoint
- `EVAL_CHECKPOINT`: Single checkpoint to evaluate
- `MIN_EXEC_TIMEOUT`: Minimum execution timeout (default: 60)

### Eval Config Structure

```json
{
  "benchmarks_folder_path": "data/new_dataset/all/eval",
  "bench_count": 64,
  "results_dir": "results/new_dataset_results/v4_7_agent",
  "json_file": "results/new_dataset_results/baselines/mlir/eval_base.json",
  "eval_runs": 1,
  "eval_aggregation": "min",
  "hardware_auto_detect": true
}
```

---

## Data Generation

### Vision Models to MLIR

```bash
# Convert ResNet18 to MLIR
python data_utils/convert/vision2mlir.py --model resnet18 --output-dir data/generated/code_files

# Convert ViT with ONNX backend
python data_utils/convert/vision2mlir.py --model vit_b_16 --backend onnx --output-dir data/generated/

# Convert with weight stripping
python data_utils/convert/vision2mlir.py --model efficientnet_b0 --strip-weights
```

**Supported models:** resnet18, resnet50, efficientnet_b0, mobilenet_v2, densenet121, vit_b_16, convnext_base, vgg16

### Transformer Models to MLIR

```bash
# Convert BERT
python data_utils/convert/transformers2mlir.py --model bert --output-dir data/generated/code_files

# Convert GPT-2
python data_utils/convert/transformers2mlir.py --model gpt2 --backend onnx

# Convert DistilBERT
python data_utils/convert/transformers2mlir.py --model distilbert
```

**Supported models:** bert, distilbert, roberta, albert, gpt2, t5, bart, lstm

### GNN Models to MLIR

```bash
# Convert GCN
python data_utils/convert/gnn2mlir.py --model gcn --output-dir data/generated/

# Convert all GNN models
python data_utils/convert/gnn2mlir.py --model all
```

**Supported models:** gcn, graphsage, gat, gin

### Extract Operations from MLIR

```bash
# Extract individual operations
python data_utils/extract/extract_ops.py \
  --input data/nn/raw/resnet18_linalg.mlir \
  --output-dir data/nn/code_files/resnet18/ \
  --batch-size 1
```

### Extract Blocks (Multi-Op Sequences)

```bash
# Extract blocks with window=5, stride=3
python data_utils/extract/extract_blocks.py \
  --input data/nn/raw_bench/resnet18_linalg.mlir \
  --output-dir data/nn/code_files/resnet18/ \
  --window 5 --stride 3
```

**Arguments:**
- `--input`: Input MLIR file
- `--output-dir`: Output directory
- `--window`: Block window size (default: 5)
- `--stride`: Stride between blocks (default: 3)
- `--max-depth`: Maximum extraction depth
- `--max-paths`: Maximum paths to extract
- `--batch-candidates`: Batch candidate size

### Unified Data Pipeline

```bash
# Build benchmark dataset
python -m data_utils.orchestrate build-benchmark \
  --input_file config.yaml --output_file data.json

# Vision model conversion
python -m data_utils.orchestrate vision \
  --model resnet50 --output-dir data/generated/

# Transformer conversion
python -m data_utils.orchestrate transformer \
  --model bert --backend onnx

# Wrap MLIR with timing
python -m data_utils.orchestrate wrap \
  --input model.mlir --model-name forward --output out.mlir

# Strip weight constants
python -m data_utils.orchestrate strip huge_model.mlir --replace
```

### Synthetic Data Generation

```bash
# Generate synthetic MLIR benchmarks
python data_utils/generate/generate_synthetic.py \
  --num-singles 100 --num-bench 50 \
  --output-dir data/all/code_files
```

---

## Baseline Measurement

### MLIR Baselines

#### Measure Baselines for All Benchmarks

```bash
python scripts/baseline/get_base.py \
  --benchmarks-dir data/new_dataset/all/train \
  --output results/baselines/train_base.json \
  --timeout 30
```

#### Measure Baselines with Config

```bash
python scripts/baseline/get_base.py --config config/v4_7.json
```

#### Parallelized Baseline Measurement (Array Job)

```bash
# 20-way parallel baseline generation
sbatch scripts/baseline/get_new_dataset_base.sh
```

#### Slurm Wrapper

```bash
sbatch scripts/baseline/get_base.sh config/v4_7.json
```

**Arguments:**
- `--benchmarks-dir`: Directory containing `.mlir` files
- `--output`: Output JSON path
- `--timeout`: Per-file timeout in seconds (default: 15)
- `--chunk-index`: Chunk index for parallelization
- `--num-chunks`: Total number of chunks

### PyTorch Baselines

#### Per-Operation PyTorch Timing

```bash
python scripts/baseline/get_pytorch_times.py \
  --benchmarks-dir data/new_dataset/all \
  --output results/pytorch_baselines.json \
  --warmup 3 --trials 10
```

#### Full Model PyTorch Timing

```bash
python scripts/baseline/get_pytorch_baselines.py \
  --models gpt2 vit_b_16 \
  --output results/pytorch_full.json
```

#### Slurm Wrapper for PyTorch Timing

```bash
sbatch scripts/baseline/get_new_dataset_pytorch.sh \
  data/new_dataset/all/eval eval_pytorch
```

### Block Baselines

```bash
# Measure block baselines
sbatch scripts/baseline/get_blocks_baseline.sh

# Fresh block baselines (no resume)
sbatch scripts/baseline/get_blocks_baseline_fresh.sh
```

---

## Reporting & Analysis

### Training Progress Reports

```bash
# Auto-detect all running experiments
python scripts/utils/report_training.py

# Specific versions
python scripts/utils/report_training.py -v v4_6 v4_7 v4_8

# Single ops dataset
python scripts/utils/report_training.py -d single_ops -v v0 v4_9_small

# Watch mode (auto-refresh every 300s)
python scripts/utils/report_training.py -w 300

# Export to JSON
python scripts/utils/report_training.py --json out.json
```

**Arguments:**
- `-v / --versions`: Version names (e.g., v4_6 v4_7)
- `-d / --dataset`: Dataset (`new` or `single_ops`)
- `-w / --watch`: Auto-refresh interval in seconds
- `-j / --jobs`: Explicit job_id:version pairs
- `-n / --trend-n`: Checkpoints for trend calculation
- `--json`: Export as JSON

### Evaluation Results Reports

```bash
# All agents
python scripts/utils/report_eval.py

# Specific agents
python scripts/utils/report_eval.py -v v4_6 v4_7

# Single ops dataset
python scripts/utils/report_eval.py -d single_ops -v v0 v4_9_small

# Best checkpoint per agent
python scripts/utils/report_eval.py --best

# Missing evaluations
python scripts/utils/report_eval.py --missing

# Specific checkpoint
python scripts/utils/report_eval.py -c 500

# Export to JSON
python scripts/utils/report_eval.py --json out.json
```

**Arguments:**
- `-v / --agents`: Agent keys
- `-d / --dataset`: Dataset (`new` or `single_ops`)
- `--missing`: Show checkpoints with models but no eval
- `--best`: Show only best checkpoint per agent
- `-c / --checkpoint`: Show specific checkpoint only
- `--json`: Export as JSON

---

## Plotting & Visualization

### Generate Paper Graphs

```bash
# Build CSVs for all 3 paper graphs
python scripts/plots/build_graph_csvs.py

# Generate PNG bar charts
python scripts/plots/plot_graphs.py

# Adjust Graph 2 for presentation
python scripts/plots/adjust_graph2.py
```

### Benchmark Classification

```bash
python scripts/plots/classify_benchmarks.py
```

### Dashboard CSVs

```bash
python scripts/generate_dashboard_csvs.py
```

---

## Full Model Optimization

### End-to-End Full Model Optimization

```bash
# Optimize full model with trained checkpoint
python scripts/full_model/optimize_full_model.py \
  --checkpoint results/v4_5_agent/models/model_715.pt \
  --output results/full_model_results.json

# Specific models
python scripts/full_model/optimize_full_model.py \
  --checkpoint model_715.pt \
  --models gcn distilbert \
  --resume
```

**Arguments:**
- `--checkpoint`: Required. Path to trained model `.pt` file
- `--models-dir`: Directory with `*_linalg.mlir` files
- `--output`: Results JSON path
- `--models`: Specific model names
- `--skip-tagging`: Skip AST dumper tagging
- `--resume`: Resume from checkpoint

### Preprocess Model (AST Dumper Tagging)

```bash
python scripts/full_model/preprocess_model.py \
  --input data/nn/raw_bench/gcn_linalg.mlir \
  --output data/nn/tagged/gcn_tagged.mlir \
  --list-tags
```

### Add Timing Wrapper

```bash
python scripts/full_model/add_timing_wrapper.py \
  --input data/nn/tagged/gcn_tagged.mlir \
  --output data/nn/wrapped/gcn_wrapped.mlir
```

### Block-Based Model Optimization

```bash
# Multi-checkpoint comparison
python scripts/full_model/optimize_model_via_blocks.py \
  --checkpoints model_700.pt model_800.pt model_1999.pt \
  --model distilbert \
  --workers 64 \
  --output results/distilbert.json

# Single checkpoint with resume
python scripts/full_model/optimize_model_via_blocks.py \
  --single-checkpoint model_1999.pt \
  --model gcn \
  --baselines-json baselines.json \
  --output results/gcn.json
```

**Arguments:**
- `--checkpoints`: Multiple `.pt` files to compare
- `--single-checkpoint`: Single `.pt` file (triggers per-block output)
- `--model`: Model name (required)
- `--workers`: Number of parallel workers
- `--window-size`: Block extraction window size
- `--stride`: Block extraction stride
- `--skip-extraction`: Skip if blocks already exist

### Slurm Array Jobs

```bash
# Full model optimization array job
sbatch --array=0-19 scripts/full_model/optimize_full_model.sh

# Block-based optimization
sbatch scripts/full_model/run_blocks_v10.sh <model_name>

# Honest block speedup
sbatch scripts/full_model/run_honest_blocks.sh
```

### Honest Model Speedup

```bash
python scripts/data/honest_blocks_speedup.py \
  --models albert bert \
  --output results/honest.json
```

---

## Utilities

### Lustre Quota Cleanup

```bash
python scripts/utils/cleanup_quota.py
```

**Warning:** Check `RUNNING_RUNS` set in the script before running.

### Migrate Results Structure

```bash
# Dry-run migration
python scripts/migrate_results.py

# Execute migration
python scripts/migrate_results.py --execute

# Single agent
python scripts/migrate_results.py -d results/v4_7_agent
```

### Migrate to Flat Structure

```bash
# Dry-run
python scripts/migrate_to_flat.py

# Execute
python scripts/migrate_to_flat.py --execute
```

### Split Baseline JSON

```bash
python scripts/data/split_json.py config/new_dataset/train/v4_7.json
```

**Arguments:**
- `--config`: Config JSON path
- `--input`: Override base exec times JSON
- `--eval-ratio`: Fraction for eval (default: 0.2)
- `--seed`: Random seed (default: 42)

### Test torch-mlir Compilation

```bash
python scripts/utils/test_torch_mlir_compile.py
```

### Submit and Monitor Job

```bash
scripts/utils/submit_and_monitor.sh scripts/train/train.sh config/v4_7.json
```

---

## Config Files Location

```
config/
├── new_dataset/
│   ├── train/           # v0, v4_5, v4_6, v4_7, v4_8, v4_9
│   └── eval/            # Matching eval configs
├── single_ops_dataset/
│   ├── train/           # v0, v4_9_small, v4_9_large
│   └── eval/            # Matching eval configs
├── old_dataset/
│   ├── train/           # Legacy configs
│   └── eval/            # Legacy eval configs
└── ablation/            # Ablation configs (no_hw, no_reward, no_transformer)
```

---

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG_FILE_PATH` | Path to config JSON | Required |
| `AUTOSCHEDULER_IMPL` | Implementation package | From config |
| `RESUME_FROM` | Resume directory path | None |
| `FORCE_NEW` | Overwrite existing results | `0` |
| `EVAL_DIR` | Override model directory | From config |
| `EVAL_START` | Start checkpoint number | `100` |
| `EVAL_END` | End checkpoint number | `10000` |
| `EVAL_STRIDE` | Checkpoint stride | `100` |
| `EVAL_LAST_ONLY` | Evaluate last checkpoint | `0` |
| `EVAL_CHECKPOINT` | Single checkpoint to evaluate | None |
| `MIN_EXEC_TIMEOUT` | Minimum execution timeout | `60` |

---

## Typical Workflow

### 1. Generate Data

```bash
# Convert models to MLIR
python data_utils/convert/vision2mlir.py --model resnet18 --output-dir data/new_dataset/nn/raw_bench/
python data_utils/convert/vision2mlir.py --model resnet50 --output-dir data/new_dataset/nn/raw_bench/

# Extract operations
python data_utils/extract/extract_ops.py \
  --input data/new_dataset/nn/raw_bench/resnet18_linalg.mlir \
  --output-dir data/new_dataset/all/

# Generate baselines
python scripts/baseline/get_base.py \
  --benchmarks-dir data/new_dataset/all/train \
  --output results/baselines/mlir/train_base.json

# Create train/eval split
python scripts/data/split_json.py config/new_dataset/train/v4_7.json
```

### 2. Train

```bash
sbatch scripts/train/train.sh config/new_dataset/train/v4_7.json
```

### 3. Evaluate

```bash
sbatch scripts/eval/eval.sh config/new_dataset/eval/v4_7_eval.json --checkpoint 500
```

### 4. Report & Plot

```bash
python scripts/utils/report_eval.py --best
python scripts/plots/build_graph_csvs.py
python scripts/plots/plot_graphs.py
```
