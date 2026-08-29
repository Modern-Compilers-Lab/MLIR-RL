# LLM Transform

Automated optimization of MLIR code using [Claude Code](https://www.anthropic.com/claude-code).

This project lets a Claude Code agent iteratively rewrite **MLIR transform schedules**, **MLIR lowering passes**, and **LLVM/llc flags** to make a kernel run as fast as possible. Each candidate configuration is compiled end-to-end (linalg → LLVM → shared library), executed, and timed against a PyTorch reference. The metric optimized is

```txt
speedup = PyTorch_time / MLIR_time
```

with a target of ≥ 2× (i.e. at least twice as fast as PyTorch). Claude interacts with the pipeline through an MCP server (`llm_transform/mcp_server.py`) that exposes three tools: `run_schedule` (compile + execute + log), `lower_schedule` (compile only, dump IR for inspection), and `get_transform_doc` (look up the documentation for a whitelisted transform op). See [resources/context.md](resources/context.md) for the full technical overview.

---

## Quick start

If all you want is to run the framework — with installation and setup taken care of for you — run a single command from the project root:

```bash
bash scripts/run.sh
```

`run.sh` verifies the environment (running `scripts/setup.sh` automatically if anything is missing), then launches an optimization session over every benchmark in [data/](data/) and prints where the results were saved. You do **not** need to do anything else.

The rest of this document covers the details — how to add your own kernels, customize sessions, read and plot results, and run individual steps by hand — if and when you want them.

## Table of contents

- [Quick start](#quick-start)
- [1. Installation](#1-installation)
- [2. Inputs](#2-inputs)
- [3. Running an optimization session](#3-running-an-optimization-session)
  - [Submitting as a Slurm job](#submitting-as-a-slurm-job)
- [4. Reading the results](#4-reading-the-results)
- [5. Plotting results](#5-plotting-results)
- [6. Running the validation tests](#6-running-the-validation-tests)
- [7. Manual single-run execution](#7-manual-single-run-execution)
- [8. Project layout](#8-project-layout)

---

## 1. Installation

You will need [Claude Code](https://docs.claude.com/en/docs/claude-code) installed and authenticated (`claude login`).

Run the setup script from the project root and follow the prompt for the conda environment name (default `llm_transform`):

```bash
bash scripts/setup.sh
```

Activating this environment is **mandatory** for every command you run inside this project, except for the bash scripts (which activate it themselves):

```bash
source scripts/env.local.sh
conda activate "$MAIN_ENV"
```

Throughout the rest of this README this two-line step is abbreviated to the comment `# <environment activation>`.

The setup script also builds the **array-dataflow equivalence verifier** (a set of custom MLIR pass plugins used by the validation harness). See the verifier's [README](llm_transform/tools/c/equivalence/README.md) for how to rebuild and run it.

## 2. Inputs

The system optimizes the MLIR files placed under [data/](data/). Each subdirectory is one benchmark, with one `.mlir` file per instance and a `sizes.json` describing problem sizes. Benchmarks currently included: `matmul`, `conv_2d`, `add`, `pooling`.

To optimize a new kernel:

1. Add `data/<name>/<instance>.mlir` with the target ops tagged `{tag = "<tag>"}` (see [resources/context.md](resources/context.md)) and an entry in `data/<name>/sizes.json`.
2. If `<name>` isn't an already existing benchmark, add a matching PyTorch reference in [llm_transform/torch_exec.py](llm_transform/torch_exec.py) (an `<name>_op` / `<name>_inputs` pair plus a `case` in `main`) and an expected-output `case` in `transform_and_run` in [llm_transform/utils/execution.py](llm_transform/utils/execution.py).

To create a new instance of an existing benchmark, use [llm_transform/tools/create_instance.py](llm_transform/tools/create_instance.py). It generates the `.mlir` file and updates `sizes.json` automatically:

```bash
# <environment activation>
python -m llm_transform.tools.create_instance <benchmark> <sizes...>
```

Refer to `data/<name>/sizes.json` for the size parameters expected by a given benchmark — pass them as `key=value` pairs (or as positional integers when the file stores a list). For example:

```bash
# <environment activation>

# key=value pairs (when sizes.json stores a dict, e.g. matmul):
python -m llm_transform.tools.create_instance matmul M=1024 K=1024 N=1024

# positional integers (when sizes.json stores a list, e.g. add):
python -m llm_transform.tools.create_instance add 64 64 64 64
```

## 3. Running an optimization session

The hands-free entry point is [scripts/run.sh](scripts/run.sh). It verifies the setup (running `scripts/setup.sh` if needed), submits the session to Slurm, waits for it to finish, and prints where the results were saved:

```bash
bash scripts/run.sh
```

With no arguments it prompts for an optional filter. To skip the prompt, pass benchmark names and/or full IDs (`<name>_<instance>`) as arguments — by default every instance in `data/` is optimized:

```bash
bash scripts/run.sh matmul_2          # just one instance
bash scripts/run.sh matmul            # every instance of one benchmark
bash scripts/run.sh matmul_2 conv_2d  # a mix of names and full IDs
```

### Submitting as a Slurm job

`run.sh` adapts to where you launch it from, so there are two ways to put the work on the cluster:

- **Run it in your current terminal session — `bash scripts/run.sh [filter ...]`.** This is the usual case. The script itself stays in your shell only to drive things: it `sbatch`'s the optimization session ([scripts/claude.sh](scripts/claude.sh)) to Slurm with `--wait`, blocks until the compute job finishes, then reports the results. The actual work runs on a compute node, so your terminal is never used for anything heavy.

- **Submit `run.sh` itself — `sbatch scripts/run.sh [filter ...]`.** Here `run.sh` *is* the Slurm job (it carries its own `#SBATCH` directives: `-p compute`, `-c 8`, `--mem=32G`, `-t 7-00`). Detecting that it is already inside an allocation, it runs the session inline in that same allocation instead of submitting a nested job. Use this if you want to detach the whole run from your terminal.

Either way the session is driven by the Slurm script [scripts/claude.sh](scripts/claude.sh) — submit it directly if you want to bypass `run.sh` entirely — and its stdout is written to `logs/claude/<JOBID>.log`.

## 4. Reading the results

Each optimization session gets its own directory under [logs/stats/](logs/stats/), keyed by `<EXPERIMENT_ID>`. All artifacts produced during the session are written there.

```txt
logs/stats/<EXPERIMENT_ID>/
  claude_optimization.log     # Append-only log of every run_schedule call (id, speedup, summary, error)
  performance.log             # Timestamped speedup samples (used by the plotting scripts)
  performance.png             # Speedup-over-time plot (auto-generated at the end of the session)
  tokens.log                  # Input/output token counts per Claude turn
  best/
    state.json                # Best speedup recorded so far per benchmark instance
    <name>/<instance>/        # Best-seen configuration for this benchmark instance:
      schedule.mlir           # Transform schedule
      passes.txt              # MLIR lowering pipeline
      llvm-llc-passes-flags.txt   # LLVM opt + llc passes/flags
  gen/                        # Output from lower_schedule (per-session to keep parallel experiments isolated)
```

`claude_optimization.log` contains everything Claude has tried during the session — to follow progress live use:

```bash
tail -f logs/stats/<EXPERIMENT_ID>/claude_optimization.log
```

Outside the per-session folders, `logs/claude/` and `logs/jobs/` hold raw Slurm stdout/stderr for the Claude job and individual execution jobs.

## 5. Plotting results

Two helper scripts under [llm_transform/tools/](llm_transform/tools/) visualize the experiment logs.

Plot speedup over time, one curve per instance, for a single experiment:

```bash
# <environment activation>
python -m llm_transform.tools.plot_performance <EXPERIMENT_ID>
```

Compare multiple experiments side by side (one subplot per instance):

```bash
# <environment activation>
python -m llm_transform.tools.plot_performance_compare <EXPERIMENT_ID_1> <EXPERIMENT_ID_2> ...
```

Both scripts read from `logs/stats/` by default and save PNGs into the corresponding stats directory.

## 6. Running the validation tests

The validation harness checks that the equivalence verifier correctly flags illegal transform schedules. Test cases live in [tests/validation/](tests/validation/) — each file contains a kernel paired with a transform schedule that is either dependence-preserving or dependence-violating.

Run the full suite from the project root:

```bash
# <environment activation>
python test_mlir_validation.py
```

Useful flags:

- `-v` / `--verbose` — print captured stderr for each test.
- `--filter <substr>` — run only tests whose filename matches the substring (e.g. `--filter tiling`).

The harness requires the equivalence verifier shared libraries built during [installation](#1-installation).

## 7. Manual single-run execution

If you want to evaluate a single configuration outside of a Claude session, use [scripts/execute.sh](scripts/execute.sh). It runs the optimized configuration and the PyTorch reference, then prints the speedup:

```bash
sbatch scripts/execute.sh -i matmul_2 \
    -t resources/base_schedule.mlir \
    -p resources/base_passes.txt
```

Pass `--id <name>_<instance>` (or `-i`) plus any flags accepted by `llm_transform/utils/execution.py` (transform schedule path, MLIR passes file, LLVM passes/flags, etc.). The base no-op schedule lives at [resources/base_schedule.mlir](resources/base_schedule.mlir) and the default lowering pipeline at [resources/base_passes.txt](resources/base_passes.txt).

## 8. Project layout

A condensed view (full layout in [resources/context.md](resources/context.md)):

```txt
data/                     # Benchmarks: <name>/<instance>.mlir + sizes.json
resources/
  base_schedule.mlir      # No-op transform schedule (starting point)
  base_passes.txt         # Default MLIR lowering pipeline
  conda/                  # Conda environment definitions
llm_transform/            # Python package (installed via `pip install -e .`)
  mcp_server.py           # MCP tools: run_schedule, lower_schedule, get_transform_doc
  torch_exec.py           # PyTorch reference execution
  utils/                  # Compilation + execution pipeline
  tools/
    plot_performance*.py  # Plotting scripts
    c/equivalence/        # Array-dataflow equivalence verifier (C++/MLIR)
pyproject.toml            # Package metadata
scripts/
  claude.sh               # Slurm: launch a Claude optimization session
  execute.sh              # Slurm: evaluate one configuration
tests/validation/         # MLIR test cases for the equivalence check
logs/                     # Experiment outputs (see section 5)
```
