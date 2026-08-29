# `llm_action/` — LLM-Synthesized RL Action Spaces for MLIR

**Phase 2 of the [MLIR-RL](../README.md) research programme.**

The bottleneck in RL for compiler optimization is not the policy — it is the **action space**. Every
action must encode legality, compose safely with the others, and expose RL-friendly parameters, all
by hand. This subproject replaces that manual effort with an **LLM agent pipeline** that enumerates,
implements, and empirically validates the action space, then trains a **MaskablePPO** agent on the
generated actions.

```
  ┌─ LLM AGENTS (Claude Code CLI + MCP) ─┐   ┌─ COMPILER ─┐   ┌─ RL ───────────┐
  │  1. Enumeration   (what actions?)     │   │ MLIR       │   │ Gymnasium env  │
  │  2. Implementation(how, in Transform) │──►│ Transform  │──►│ MultiDiscrete  │
  │  3. Exploration   (do they compose?)  │   │ + real     │   │ MaskablePPO    │
  └───────────────────────────────────────┘   │  execution │   │ reward=speedup │
                                              └────────────┘   └────────────────┘
```

> **Two documents, two purposes.**
> This README is the **operating manual** — how to set up, what to run, what comes out.
> [DOCUMENTATION.md](DOCUMENTATION.md) is the **internals dossier** — the action contract, the MDP
> formulation, the observation encoding, the reward design, with figures. Read this file first, then
> that one when you need to understand or modify the internals.

---

## Table of contents

1. [Setup](#1-setup)
2. [Layout](#2-layout)
3. [Datasets](#3-datasets)
4. [Action-space versions](#4-action-space-versions)
5. [Workflow A — generate an action space](#workflow-a--generate-an-action-space)
6. [Workflow B — train PPO](#workflow-b--train-ppo)
7. [Workflow C — evaluate](#workflow-c--evaluate)
8. [Workflow D — one-off execution and baselines](#workflow-d--one-off-execution-and-baselines)
9. [The MCP servers](#9-the-mcp-servers)
10. [Troubleshooting and gotchas](#10-troubleshooting-and-gotchas)

---

## 1. Setup

### Environment

```bash
conda activate mlir
```

This environment is known-good: Python 3.14, MLIR/LLVM 21 with Python bindings, `torch` 2.10 (cpu),
`gymnasium`, `stable-baselines3` + `sb3_contrib`, `fastmcp`, `dask` + `dask_jobqueue`, `wandb`,
`anthropic`. It can be recreated from [../env/mlir-env.yml](../env/mlir-env.yml). No LLVM source
build is required — the conda packages provide `mlir-opt`, `llc`, the Python bindings, and the
runtime shared libraries.

### Always run from the repository root

Everything here is imported as `llm_action.*`, so the working directory must be the repo root:

```bash
cd /scratch/kb5213/workspace/MLIR-RL
python -m llm_action.src.rl.train_ppo --help
```

This is also why [../.mcp.json](../.mcp.json) pins `"cwd": "/scratch/kb5213/workspace/MLIR-RL"`, and
why every `sbatch` line below is written as `sbatch llm_action/scripts/<script>.sh` — from the root,
not from inside `llm_action/`.

### Paths and credentials

`PROJECT_ROOT` is **hard-coded** in [src/config.py](src/config.py):

```python
PROJECT_ROOT = Path("/scratch/kb5213/workspace/MLIR-RL/")
```

Every derived path (`DATA_DIR`, `RL_RESULTS_DIR`, the SLURM script paths, the Dask temp dir) hangs
off it. **This is the first thing to change if you clone the repo elsewhere.**

Then copy the credentials template:

```bash
cp llm_action/.env.example llm_action/.env   # then fill it in
```

- `ANTHROPIC_API_KEY` — for the SDK-based agents in [src/agents/](src/agents/). The Claude Code CLI
  used by the pipeline scripts authenticates separately (`claude login`).
- `AST_DUMPER_BIN_PATH` — **required for training and evaluation**. The observation encoder
  ([src/env/state_extractor.py](src/env/state_extractor.py)) shells out to this binary to dump the
  tagged operation's loop bounds and access patterns. Build it from
  [../tools/ast_dumper/](../tools/ast_dumper/) (see [../rl_autoschedular/README.md](../rl_autoschedular/README.md#building-the-c-tools)).
- `MLIR_SHARED_LIBS` — comma-separated `libomp.so,libmlir_c_runner_utils.so,libmlir_runner_utils.so`.
  With the `mlir` conda env these live in `$CONDA_PREFIX/lib`; the helper
  [../scripts/setup_env.sh](../scripts/setup_env.sh) sets it for you.

## 2. Layout

| Path | Contents |
|---|---|
| [src/actions/](src/actions/) | `base.py` (the `ActionBase` contract), `test.py` (standalone test harness), and one directory per generated action-space version `v0 … v55` |
| [src/agents/](src/agents/) | SDK-based agent wrappers (`action_enumeration`, `action_implementation`, `documentation_lookup`, `optimization`, `parametrizer`) |
| [src/prompts/](src/prompts/) | Prompt generators. `claude_*.py` build the CLI prompt for each pipeline layer; `action_enumeration.py` / `action_implementation.py` / `schedule_exploration.py` render the methodology specs |
| [src/env/](src/env/) | The Gymnasium MDP: `mlir_opt_env.py`, `action_space.py`, `action_registry.py`, `state_extractor.py`, `env_config.py`, `benchmarks.py` |
| [src/rl/](src/rl/) | `train_ppo.py`, `evaluate_ppo.py`, `behavior_masking.py` |
| [src/mcp/](src/mcp/) | `mcp_server.py` (full), `mcp_server_minimal.py`, `utils.py` (SLURM submit/poll) |
| [src/execution/](src/execution/) | `mlir_execution.py`, `torch_execution.py`, and the `local` / `dask` / `slurm` executors |
| [src/utils/](src/utils/) | `transformation.py` (the default bufferization + lowering pipeline), parsing, caching, scraping |
| [src/config.py](src/config.py) | All global constants: paths, timeouts, observation and action-space dimensions |
| [resources/prompts/v1/](resources/prompts/v1/) | The versioned **methodology specs** — the agents' system prompts |
| [resources/ready/](resources/ready/) | Pre-scraped Transform-dialect documentation used by the lookup agent |
| [scripts/](scripts/) | SLURM entry points (see the workflows below) |
| [data/benchmarks/](data/benchmarks/) | The kernel datasets |
| [results/](results/) | `rl/<run>/` training runs, `evaluation/` evaluation outputs, plus per-layer agent outputs |
| [docs/](docs/) | MCP catalogues, the hardware/optimization reference, pipeline cost metrics |
| [playground/](playground/) | Comparison plots and the script that generates them |

## 3. Datasets

Under [data/benchmarks/](data/benchmarks/), one directory per operator family, each split into
`train/` and `eval/`. Counts as they stand on disk:

| Set | Operator | train | eval |
|---|---|--:|--:|
| [dataset_matmul](data/benchmarks/dataset_matmul/) | `linalg.matmul` | 187 | 14 |
| [dataset_conv2d](data/benchmarks/dataset_conv2d/) | `linalg.conv_2d_nchw_fchw` | 278 | 18 |
| [dataset_pooling](data/benchmarks/dataset_pooling/) | `linalg.pooling_nchw_max` | 250 | 10 |
| [dataset_add](data/benchmarks/dataset_add/) | elementwise add (4-D) | 271 | 10 |
| [dataset_relu](data/benchmarks/dataset_relu/) | elementwise ReLU | 149 | 14 |
| [dataset_ml](data/benchmarks/dataset_ml/) | mixed / composite ML graphs | 1135 | 66 |

Conventions:

- **Every kernel carries `attributes {tag = "operation_0"}`** on the operation to optimize. This is
  the targeting contract — actions match it with
  `transform.structured.match ... attributes{tag = "operation_0"}`, never by position or text
  rewriting, and must re-annotate the result so the next action in the schedule can find it.
- [data/benchmarks/templates/](data/benchmarks/templates/) holds the parametric `.mlir` templates the
  concrete instances are generated from.
- `baselines.json` in each set caches measured unoptimized-MLIR and PyTorch times, so training does
  not re-measure them every episode.
- [data/tensor/](data/tensor/) holds a handful of standalone kernels used for manual experiments.

## 4. Action-space versions

Each pipeline run produces a **new, independent** directory `src/actions/v<x>/`. Versions `v0`
through `v55` exist. The agents are forbidden from reading any version other than `v0` (the empty
structural skeleton), so each synthesis is unbiased by the previous one.

A completed version contains:

```
src/actions/v48/
  enumeration/action_enumeration.json   # layer 1 output: intents -> macro transformations
  enumeration/reasoning.md
  implementation/<action>.py            # layer 2 output: one ActionBase subclass per action
  tests/test_<action>.py                # layer 2 output: end-to-end test per action
  registry.py                           # ACTION_CLASSES + ACTION_DEPENDENCIES + SCHEDULE_GRAPH
  mcp.py                                # per-action MCP server (consumed by layer 3)
```

The curated, paper-relevant versions pair one family with one action space:

| Version | Family | Actions |
|---|---|---|
| `v48` | matmul | `Tiling`, `LoopInterchange`, `VectorizationSequential`, `VectorizationParallel`, `ParallelizationTiling`, `ParallelizationThreads` |
| `v49` | conv2d | + `Promotion`, `Im2colLowering` (8 actions) |
| `v50` | pooling | `Tiling`, `LoopInterchange`, `SequentialVectorization`, `ParallelVectorization`, `TilingParallelization`, `ThreadParallelization` |
| `v51` | add | `Tiling`, `LoopInterchange`, `VectorizationSeq`, `VectorizationPar`, `ParallelizationTile`, `ParallelizationThreads` |
| `v52` | relu | as `v48` |
| `v53`, `v55` | ML / mixed | later runs |

Note that the agent chooses its own action names per run — `VectorizationSequential` in `v48` is
`VectorizationSeq` in `v51` and `SequentialVectorization` in `v50`. Always read the version's own
`registry.py`.

`registry.py` carries the three artefacts the environment consumes:

- `ACTION_CLASSES` — the action set (layer 2).
- `ACTION_DEPENDENCIES` — a **denylist**: "once X has run, Y is illegal" (layer 3). For `v48`, both
  vectorization actions block everything, because vectorization consumes the `linalg` op — lowering
  is terminal.
- `SCHEDULE_GRAPH` — a per-family **allowlist** of empirically good schedule *shapes* (layer 3).
  For `v48` matmul:

  ```python
  SCHEDULE_GRAPH = {"matmul": [
      ["VectorizationParallel"],
      ["ParallelizationThreads", "VectorizationSequential"],
      ["ParallelizationTiling",  "VectorizationSequential"],
      ["ParallelizationTiling",  "Tiling", "VectorizationSequential"],
      ["ParallelizationTiling",  "LoopInterchange", "VectorizationParallel"],
  ]}
  ```

  The policy navigates this tree and still tunes every action's parameters itself. Layer 3
  deliberately discovers *shapes*, never parameters — parameter tuning is the RL policy's job.

---

## Workflow A — generate an action space

Three SLURM jobs, run in order. Each launches one Claude Code CLI session whose prompt is generated
by a Python script that embeds the methodology spec plus a representation of the target benchmark
family.

```bash
claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_<layer>.py $ARGS)"
```

### Layer 1 — Enumeration (*what* actions should exist)

```bash
sbatch llm_action/scripts/claude_enumeration.sh --benchmark dataset_matmul --limit 5
```

| | |
|---|---|
| Flags | `--benchmark <set>` (default `standard`), `--limit <n>` (how many sibling kernel shapes to list in the prompt — a token-budget knob) |
| Spec | [resources/prompts/v1/action_enumeration.md](resources/prompts/v1/action_enumeration.md) |
| Agent role | *Expert MLIR Optimization Engineer* — reasons abstractly about the loop nest; writes no code |
| Writes | `src/actions/v<x>/enumeration/action_enumeration.json` + `reasoning.md` |
| Logs | `logs/jobs/claude/enum_<jobid>.{out,err}` |

The spec itself is parameterized. Regenerate it before the run to change how many optimization
intents and transformations the agent must produce:

```bash
python llm_action/src/prompts/action_enumeration.py \
    --intents_num_min 2 --intents_num_max 3 \
    --transformations_num_min 2 --transformations_num_max 4
```

### Layer 2 — Implementation (*how*, in the Transform dialect)

```bash
sbatch llm_action/scripts/claude_implementation.sh --benchmark dataset_matmul --limit 5
```

| | |
|---|---|
| Flags | `--benchmark`, `--limit` |
| Spec | [resources/prompts/v1/action_implementation.md](resources/prompts/v1/action_implementation.md) |
| Agent role | *Expert MLIR Transformation Engineer* — turns one abstract transformation into one executable action |
| Needs | the **full** MCP server (`mlir-tools` → `src.mcp.mcp_server`) — it needs `transform_mlir_code` and `delegate_documentation_lookup` |
| Writes | `implementation/<action>.py`, `tests/test_<action>.py`, `registry.py`, `mcp.py` |
| Logs | `logs/jobs/claude/implem_<jobid>.{out,err}` |

The agent self-validates every action with a fixed 5-step tool plan per kernel: look up the
Transform-dialect docs → execute the original → transform → execute the transformed → measure
speedup. Non-obvious MLIR quirks it discovers are appended to the scratchpad
[docs/memory/MEMORY.md](docs/memory/MEMORY.md) so later runs do not rediscover them.

Test any generated action standalone:

```bash
python -m llm_action.src.actions.v48.tests.test_tiling
```

### Layer 3 — Schedule exploration (*do they compose?*)

```bash
sbatch llm_action/scripts/claude_exploration.sh --action-version v48 --benchmark dataset_matmul --limit 20
```

| | |
|---|---|
| Flags | `--action-version v<x>` (required), `--benchmark` / `--benchmarks-name`, `--limit` |
| Spec | [resources/prompts/v1/schedule_exploration.md](resources/prompts/v1/schedule_exploration.md) |
| Agent role | *Expert MLIR Schedule Exploration Engineer* — treats the action set as a black box |
| Needs | the **minimal** MCP server + the per-action `rl-action-v<x>` server |
| Writes | appends `SCHEDULE_GRAPH` and `ACTION_DEPENDENCIES` to `src/actions/v<x>/registry.py` |
| Logs | `logs/jobs/claude/explore_<jobid>.{out,err}` and readable transcripts in `logs/mcp/v<x>/<kernel>_<datetime>.md` |

Phases: measure baselines → apply each action alone → **exhaustive** ordered-pair composability
matrix (every cell from a real tool call) → chain 3+ actions into schedule skeletons → pick the best
shape per kernel subset. Phases 3–4 spawn parallel sub-agents, one per kernel subset.

> **Before running layer 3, update [../.mcp.json](../.mcp.json).** Add the new per-action server and
> switch the base server to the *minimal* variant, so the agent cannot bypass the generated actions
> by writing raw Transform IR:
>
> ```json
> {"mcpServers": {
>   "mlir-tools":    {"command": "conda", "args": ["run","-n","mlir","--no-capture-output","python","-m","llm_action.src.mcp.mcp_server_minimal"], "cwd": "/scratch/kb5213/workspace/MLIR-RL"},
>   "rl-action-v48": {"command": "conda", "args": ["run","-n","mlir","--no-capture-output","python","-m","llm_action.src.actions.v48.mcp"],       "cwd": "/scratch/kb5213/workspace/MLIR-RL"}
> }}
> ```
>
> The file currently points at `v55`. `claude_exploration.sh` and `claude_implementation.sh` run
> `claude /mcp` first to connect.

### What a pipeline run costs

Measured over the v48–v52 runs ([docs/llm_pipeline_metrics.md](docs/llm_pipeline_metrics.md)):

| Family | Total output tokens | Active time | Tool calls |
|---|--:|--:|--:|
| matmul (v48) | 84,426 | 43 min | 336 |
| conv2d (v49) | 139,728 | 2 h 23 min | 788 |
| pooling (v50) | 75,265 | 55 min | 367 |
| add (v51) | 99,488 | 55 min | 421 |
| relu (v52) | 82,056 | 54 min | 322 |

Exploration dominates — roughly two thirds of the tokens and time, because every cell of the
composability matrix is a real compile-and-execute.

---

## Workflow B — train PPO

```bash
sbatch llm_action/scripts/train.sh \
    --action-version v48 \
    --benchmarks-name dataset_matmul \
    --exp-name v48_dataset_matmul_graph \
    --masking-mode schedule_graph \
    --max-steps 4
```

Runs `python -m llm_action.src.rl.train_ppo`. `MaskablePPO` (`sb3_contrib`) over an MLP actor-critic,
with a `MultiDiscrete` action space: dimension 0 selects the transformation, the remaining dimensions
are the concatenated parameter slots of every action.

### The flags that matter

| Flag | Default | Notes |
|---|---|---|
| `--action-version` | `v10` | Which generated action set to load |
| `--benchmarks-name` | `standard` | Directory under `data/benchmarks/` — **there is no `standard` set on disk, always pass this explicitly** |
| `--masking-mode` | `dependencies` | `dependencies` (denylist) · `schedule_graph` (allowlist) · `none`. **This is the central experimental knob** |
| `--max-steps` | 7 | Transformation steps per episode. Recorded runs use 4 (matmul/add/relu) and 5 (conv2d/pooling) |
| `--reward-mode` | `final` | `final` (episode end) · `intermediate` (vs previous step) · `schedule` (vs original baseline) |
| `--reward-scale` | `log` | `log` · `raw` · `delta` · `relative` |
| `--reward-baseline` | `mlir` | `mlir` (unoptimized) or `torch` (beat-PyTorch framing) |
| `--ent-coef` / `--ent-coef-final` | 5e-3 / 5e-5 | Annealed exploration; `--ent-coef-schedule linear\|exponential`, `--ent-coef-decay-frac` |
| `--net-arch` | `128 128` | Larger (e.g. `256 256 256`) for conv2d / ml |
| `--total-timesteps` | 125000 | |
| `--n-steps` / `--batch-size` / `--n-epochs` | 128 / 32 / 8 | Standard PPO rollout and update sizes |
| `--lr` / `--gamma` / `--gae-lambda` / `--clip-range` | 3e-4 / 0.99 / 0.95 / 0.2 | |
| `--vf-coef` / `--max-grad-norm` | 0.05 / 0.5 | |
| `--executor-type` | `dask` | `local` (in-process bindings) · `dask` (persistent workers) · `slurm` (one job per step). Set `DASK_NODES` for `dask` |
| `--param-mode` | `multidiscrete` | `two_policy` and `llm` are experimental alternatives |
| `--history-mode` | `success-encoding` | How past actions are encoded in the observation |
| `--loop-bound-encoding` | `log` | `log` or `max` normalization of loop bounds |
| `--policy-mask-mode` | `hard` | `behavior-only` uses the unmasked distribution for log-probs (see [src/rl/behavior_masking.py](src/rl/behavior_masking.py)) |
| `--exp-name` / `-n` | — | Free-form label appended to the run name |
| `--seed` | 42 | |
| `--resume` | — | Path to a checkpoint `.zip` |
| `--wandb-project` / `--wandb-entity` | `mlir-rl` / — | |
| `--checkpoint-freq` / `--eval-freq` | 10000 / 2000 | |

`--help` on `python -m llm_action.src.rl.train_ppo` lists the complete set; the environment-side
defaults are the `EnvConfig` dataclass in [src/env/env_config.py](src/env/env_config.py) (including
the penalties `failed_transform_penalty=-1.0`, `failed_exec_penalty=-5.0`, `no_action_penalty=-0.1`
and `max_speedup_cap=1000.0`).

### Free vs graph — the main experiment

Most runs in [results/rl/](results/rl/) come in pairs that differ only in `--masking-mode`:

- **free** (`--masking-mode none`, or `dependencies`) — the policy explores the whole action product
  space, pruned only by preconditions and structural blocks. Broad; can find novel schedules; slower
  to converge.
- **graph** (`--masking-mode schedule_graph`) — the policy may only extend the applied prefix along a
  `SCHEDULE_GRAPH` path. Narrow; high sample efficiency; learns parameters inside known-good shapes.

This is the payoff of layer 3: its empirical exploration becomes a learned prior that shrinks the RL
search space.

The worked `sbatch` lines for all five families, free and graph, are kept at the bottom of
[scripts/train.sh](scripts/train.sh).

### Outputs

Runs land in `results/rl/ppo_<YYYYMMDD_HHMMSS>_<version>_<dataset>_<exp-name>/`:

```
config.json            # every resolved hyperparameter — the reproducibility record
progress.csv           # SB3 logger: rewards, losses, entropy per update
per_benchmark_evals/   # periodic per-kernel evaluation JSON
best_model/            # best checkpoint by evaluation speedup
checkpoints/           # periodic checkpoints
final_model.zip
events.out.tfevents.*  # tensorboard
wandb/ , wandb_run_id.txt
```

`results/rl/` is gitignored — treat it as scratch and copy anything worth keeping into
`results/training/`.

---

## Workflow C — evaluate

```bash
sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260529_191812_v48_dataset_matmul_graph
```

Runs `python -m llm_action.src.rl.evaluate_ppo`. Two modes:

| Flag | Default | Notes |
|---|---|---|
| `--run-name` | **required** | Directory name under `results/rl/` |
| `--mode` | `execution` | `execution` re-runs the best model against the eval split with real timing; `training-logs` reconstructs the best evaluation from the training JSON logs — fast, no compute |
| `--eval-greedy-runs` | 9 | Greedy rollouts per kernel |
| `--enable-sampling` / `--no-enable-sampling` | off | Also do stochastic rollouts |
| `--eval-sample-runs` | 5 | Stochastic rollouts per kernel when sampling is on |
| `--executor-type` / `--dask-nodes` | `dask` / 1 | |
| `--out-name` | — | Override the output directory name |
| `--seed` | 42 | |

Outputs go to `results/evaluation/auto_action_mlir_rl/<run-name>/`:

- `per_kernel.csv` — one row per evaluation kernel (`exec_time_ms`, `speedup`, `speedup_to_torch`, …)
- `summary.csv` — aggregate per operator family and overall
- `results.json` — the full record including the chosen schedules

`summary.csv` schema and one **recorded** result, from the v48 matmul graph-schedule run
(`ppo_20260529_191812_v48_dataset_matmul_graph`, 14 eval kernels):

```csv
group,n_kernels,mean_speedup,geomean_speedup,mean_exec_time_ms,mean_speedup_to_torch,geomean_speedup_to_torch
matmul,14,403.23,348.83,3.03,1.28,1.11
```

That is ~349× (geomean) over the unoptimized MLIR baseline and ~1.11× (geomean) over PyTorch on the
same shapes and hardware. This is a measurement from one specific run, not a general claim — always
re-measure for your own configuration.

Comparison plots against the phase-1 system and PyTorch are produced by
[playground/evaluation/scripts/plot_comparison.py](playground/evaluation/scripts/plot_comparison.py)
into [playground/evaluation/plots/](playground/evaluation/plots/).

---

## Workflow D — one-off execution and baselines

Useful for sanity checks and for measuring a schedule by hand.

**Run an MLIR file** (median of several timed runs, on a reserved node with correct thread affinity):

```bash
sbatch llm_action/scripts/mlir.sh llm_action/data/tensor/matmul/matmul_128_256_128.mlir
sbatch llm_action/scripts/mlir.sh <code_file> --transform-file <schedule.mlir>
sbatch llm_action/scripts/mlir.sh <code_file> --pass-pipeline pass1 pass2 pass3
```

`--transform-file` overrides the default bufferization + vector-lowering transform;
`--pass-pipeline` overrides the default LLVM lowering pipeline. Both defaults live in
[src/utils/transformation.py](src/utils/transformation.py). Sample schedules and pipelines are in
[resources/samples/](resources/samples/).

**Measure the PyTorch reference** for a shape:

```bash
sbatch llm_action/scripts/torch.sh matmul 256 256 512
sbatch llm_action/scripts/torch.sh conv2d 128 32 7 7 256 1 1 7 7      # N C H W F KH KW OH OW
sbatch llm_action/scripts/torch.sh add 112 112 120 150
sbatch llm_action/scripts/torch.sh pooling_nchw_max 128 128 112 112 1 1 56 56
sbatch llm_action/scripts/torch.sh relu 128 128 56 56
```

All ops accept `--dtype float64`, `--fill-value`, `--warmup-iters`, `--bench-iters`. Because
`torch.sh` and `mlir.sh` carry identical `#SBATCH` resources and thread-affinity settings, the
speedup comparison is apples-to-apples.

---

## 9. The MCP servers

The MCP servers are how the LLM agents *act on* and *measure* MLIR. Full catalogues:
[docs/MCP.md](docs/MCP.md) and [docs/MCP_MINIMAL.md](docs/MCP_MINIMAL.md).

| Server | Module | Tools |
|---|---|---|
| `mlir-tools` (full) | `llm_action.src.mcp.mcp_server` | `transform_mlir_code`, `execute_mlir_code`, `execute_torch_{matmul,conv2d,add,pooling_nchw_max,relu}_by_shape`, `measure_speedup`, `delegate_documentation_lookup` |
| `mlir-tools` (minimal) | `llm_action.src.mcp.mcp_server_minimal` | the same minus `transform_mlir_code` and the doc lookup; stricter `measure_speedup` |
| `rl-action-v<x>` | `llm_action.src.actions.v<x>.mcp` | one tool per generated action: `(code, params) -> (precondition, transformed, postcondition)` |

Two things worth knowing:

- The `execute_*` tools **submit SLURM jobs internally** ([src/mcp/utils.py](src/mcp/utils.py):
  submit → poll `squeue` every `SLURM_POLL_INTERVAL=2 s` up to `SLURM_TIMEOUT=300 s` → parse the JSON
  result). MLIR and PyTorch therefore run on identical reserved hardware.
- `delegate_documentation_lookup` is a retrieval sub-agent over the scraped Transform-dialect docs in
  [resources/ready/](resources/ready/), so the synthesizing agent grounds op names and handle types
  instead of hallucinating them.

Connect from a Claude session with `claude /mcp`.

## 10. Troubleshooting and gotchas

**Setup**

- `ModuleNotFoundError: llm_action` → you are not in the repo root. `cd /scratch/kb5213/workspace/MLIR-RL`.
- Paths point somewhere unexpected → `PROJECT_ROOT` in [src/config.py](src/config.py) is hard-coded.
- `--benchmarks-name` defaults to `standard`, which **does not exist** under `data/benchmarks/`.
  Always pass a real set name.
- State extraction fails or times out → `AST_DUMPER_BIN_PATH` is unset or the binary is missing
  (`AST_DUMPER_TIMEOUT = 40 s`).

**Measurement**

- Timings slow, noisy, or unreproducible → the thread-affinity block is missing from your job script.
  See [the root README](../README.md#thread-affinity--mandatory-for-valid-timings). This fails
  silently; suspect it first.
- `AssocGrpCpuLimit` job rejection → add `#SBATCH --qos=c2`. It is present in `mlir.sh`, `torch.sh`,
  `train.sh` and `evaluate.sh`, but **not** in the `claude_*.sh` scripts.
- Timeouts to tune in [src/config.py](src/config.py): `CODE_TRANSFORM_TIMEOUT=60`,
  `CODE_EXECUTION_TIMEOUT=60`, `SLURM_TIMEOUT=300`, `DASK_WAIT_TIMEOUT=300`. The Dask outer timeout
  must stay above the inner bindings budget.

**Transformations**

- Vectorization silently fails or produces slow code → the safety envelope is: total vector size
  ≤ `VECTORIZATION_SIZE_LIMIT` (2048) elements, rank ≤ 3, and **sizes must divide the loop bounds**.
  A non-dividing size produces a dynamic remainder loop that downstream lowering cannot vectorize.
- An action after vectorization does nothing → vectorization is **terminal**. It consumes the
  `linalg` op; the tag moves to the outermost generated `scf.for`, and no structure-preserving action
  can follow. This is why every `SCHEDULE_GRAPH` path ends in a vectorization action.
- The tag disappeared mid-schedule → an action failed to re-annotate `tag = "operation_0"`. The
  episode terminates. See the tagging contract in [DOCUMENTATION.md](DOCUMENTATION.md) §A3.2.

**Agents**

- Layer 3 writes raw Transform IR instead of using the actions → you left the *full* MCP server
  connected. Switch `mlir-tools` to `mcp_server_minimal` in [../.mcp.json](../.mcp.json).
- Layer 2 invents Transform op names → check that `delegate_documentation_lookup` is reachable and
  that [resources/ready/](resources/ready/) is populated.
