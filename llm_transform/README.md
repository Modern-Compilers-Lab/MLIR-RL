# LLM Transform

Automated optimization of MLIR code using [Claude Code](https://www.anthropic.com/claude-code).

This project lets a Claude Code agent iteratively rewrite **MLIR transform schedules**, **MLIR lowering passes**, and **LLVM/llc flags** to make a kernel run as fast as possible. Each candidate configuration is compiled end-to-end (linalg → LLVM → shared library), executed, and timed against a PyTorch reference. The metric optimized is

```txt
speedup = PyTorch_time / MLIR_time
```

with a target of ≥ 2× (i.e. at least twice as fast as PyTorch). Claude interacts with the pipeline through an MCP server (`src/mcp_server.py`) that exposes two tools, `run_schedule` (compile + execute + log) and `lower_schedule` (compile only, dump IR for inspection). See [CLAUDE.md](CLAUDE.md) for the full technical overview.

---

## 1. Installation

Two conda environments are needed: `main` (MLIR + Claude pipeline) and `torch-cpu` (PyTorch reference). Both definitions live in [resources/conda/](resources/conda/).

```bash
conda env create -f resources/conda/main.yml
conda env create -f resources/conda/torch-cpu.yml
```

Activate the main environment for everything except the PyTorch reference run:

```bash
conda activate main
```

You will also need [Claude Code](https://docs.claude.com/en/docs/claude-code) installed and authenticated (`claude login`).

## 2. Building the legality check pass

The polyhedral legality check is a custom MLIR pass built as a shared library plugin. Build it once after installing the conda environment:

```bash
conda activate main
cd src/tools/c/dependence
make
```

This produces `src/tools/c/dependence/build/lib/libPolyhedralLegalityCheck.so`, which is loaded by the validation harness and the MCP server. To rebuild from scratch use `make clean && make`.

## 3. Inputs

The system optimizes the MLIR files placed under [data/](data/). Each subdirectory is one benchmark, with one `.mlir` file per instance and a `sizes.json` describing problem sizes. Benchmarks currently included: `matmul`, `conv_2d`, `add`, `pooling`.

To optimize a new kernel:

1. Add `data/<name>/<instance>.mlir` with the target ops tagged `{tag = "<tag>"}` (see [CLAUDE.md](CLAUDE.md)) and an entry in `data/<name>/sizes.json`.
2. If `<name>` is not one of the benchmarks already handled in [src/torch_exec.py](src/torch_exec.py), add a matching PyTorch reference there (an `<name>_op` / `<name>_inputs` pair plus a `case` in `main`) so the speedup metric can be computed against PyTorch.

## 4. Running an optimization session

The entry point is the Slurm script [scripts/claude.sh](scripts/claude.sh). Submit it from the project root:

```bash
sbatch scripts/claude.sh
```

Slurm stdout for the Claude session is written to `logs/claude/<JOBID>.log`.

## 5. Reading the results

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

## 6. Plotting results

Two helper scripts under [src/tools/](src/tools/) visualize the experiment logs.

Plot speedup over time, one curve per benchmark, for a single experiment:

```bash
python src/tools/plot_performance.py <EXPERIMENT_ID>
```

Compare multiple experiments side by side (one subplot per benchmark):

```bash
python src/tools/plot_performance_compare.py <EXPERIMENT_ID_1> <EXPERIMENT_ID_2> ...
```

Both scripts read from `logs/stats/` by default and save PNGs into the corresponding stats directory.

## 7. Running the validation tests

The validation harness checks that the polyhedral legality pass correctly flags illegal transform schedules. Test cases live in [tests/validation/](tests/validation/) — each file contains a kernel paired with a transform schedule that is either dependence-preserving or dependence-violating.

Run the full suite from the project root:

```bash
conda activate main
python test_mlir_validation.py
```

Useful flags:

- `-v` / `--verbose` — print captured stderr for each test.
- `--filter <substr>` — run only tests whose filename matches the substring (e.g. `--filter tiling`).

The harness requires the legality pass shared library from [step 2](#2-building-the-legality-check-pass).

## 8. Manual single-run execution

If you want to evaluate a single configuration outside of a Claude session, use [scripts/execute.sh](scripts/execute.sh). It runs the optimized configuration and the PyTorch reference, then prints the speedup:

```bash
sbatch scripts/execute.sh -i matmul_2 \
    -t resources/base_schedule.mlir \
    -p resources/base_passes.txt
```

Pass `--id <name>_<instance>` (or `-i`) plus any flags accepted by `src/utils/execution.py` (transform schedule path, MLIR passes file, LLVM passes/flags, etc.). The base no-op schedule lives at [resources/base_schedule.mlir](resources/base_schedule.mlir) and the default lowering pipeline at [resources/base_passes.txt](resources/base_passes.txt).

## 9. Project layout

A condensed view (full layout in [CLAUDE.md](CLAUDE.md)):

```txt
data/                     # Benchmarks: <name>/<instance>.mlir + sizes.json
resources/
  base_schedule.mlir      # No-op transform schedule (starting point)
  base_passes.txt         # Default MLIR lowering pipeline
  conda/                  # Conda environment definitions
src/
  mcp_server.py           # MCP tools: run_schedule, lower_schedule
  torch_exec.py           # PyTorch reference execution
  utils/                  # Compilation + execution pipeline
  tools/
    plot_performance*.py  # Plotting scripts
    c/dependence/         # Polyhedral legality check pass (C++/MLIR)
scripts/
  claude.sh               # Slurm: launch a Claude optimization session
  execute.sh              # Slurm: evaluate one configuration
tests/validation/         # MLIR test cases for the legality check
logs/                     # Experiment outputs (see section 5)
```
