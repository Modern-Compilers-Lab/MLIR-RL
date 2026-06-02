# LLM Transform

Automated optimization of MLIR code using [Claude Code](https://www.anthropic.com/claude-code).

This project lets a Claude Code agent iteratively rewrite **MLIR transform schedules**, **MLIR lowering passes**, and **LLVM/llc flags** to make a kernel run as fast as possible. Each candidate configuration is compiled end-to-end (linalg → LLVM → shared library), executed, and timed against a PyTorch reference. The metric optimized is

```txt
speedup = PyTorch_time / MLIR_time
```

with a target of ≥ 2× (i.e. at least twice as fast as PyTorch). Claude interacts with the pipeline through an MCP server (`llm_transform/mcp_server.py`) that exposes two tools, `run_schedule` (compile + execute + log) and `lower_schedule` (compile only, dump IR for inspection). See [resources/context.md](resources/context.md) for the full technical overview.

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

Then install this project as an editable Python package so `llm_transform.*` imports resolve from anywhere:

```bash
pip install -e .
```

You will also need [Claude Code](https://docs.claude.com/en/docs/claude-code) installed and authenticated (`claude login`).

## 2. Building the equivalence verifier

The **array-dataflow equivalence verifier** is a set of custom MLIR passes, built as shared library plugins, that prove a transform schedule preserves a kernel's semantics. Build them once after installing the conda environment:

```bash
conda activate main
cd llm_transform/tools/c/equivalence
make
```

This produces the plugins under `llm_transform/tools/c/equivalence/build/lib/` (`libEquivalenceVerifier.so`, `libTagLinalgOps.so`, `libRaiseSCFToAffine.so`), which are loaded by the validation harness. To rebuild from scratch use `make clean && make`. See the verifier's [README](llm_transform/tools/c/equivalence/README.md) for installation details and the different ways to run it.

## 3. Inputs

The system optimizes the MLIR files placed under [data/](data/). Each subdirectory is one benchmark, with one `.mlir` file per instance and a `sizes.json` describing problem sizes. Benchmarks currently included: `matmul`, `conv_2d`, `add`, `pooling`.

To optimize a new kernel:

1. Add `data/<name>/<instance>.mlir` with the target ops tagged `{tag = "<tag>"}` (see [resources/context.md](resources/context.md)) and an entry in `data/<name>/sizes.json`.
2. If `<name>` isn't an already existing benchmark, add a matching PyTorch reference in [llm_transform/torch_exec.py](llm_transform/torch_exec.py) (an `<name>_op` / `<name>_inputs` pair plus a `case` in `main`) and an expected-output `case` in `transform_and_run` in [llm_transform/utils/execution.py](llm_transform/utils/execution.py).

To create a new instance of an existing benchmark, use [llm_transform/tools/create_instance.py](llm_transform/tools/create_instance.py). It generates the `.mlir` file and updates `sizes.json` automatically:

```bash
python -m llm_transform.tools.create_instance <benchmark> <sizes...>
```

Refer to `data/<name>/sizes.json` for the size parameters expected by a given benchmark — pass them as `key=value` pairs (or as positional integers when the file stores a list). For example:

```bash
# key=value pairs (when sizes.json stores a dict, e.g. matmul):
python -m llm_transform.tools.create_instance matmul M=1024 K=1024 N=1024

# positional integers (when sizes.json stores a list, e.g. add):
python -m llm_transform.tools.create_instance add 64 64 64 64
```

## 4. Running an optimization session

The entry point is the Slurm script [scripts/claude.sh](scripts/claude.sh). Submit it from the project root:

```bash
sbatch scripts/claude.sh
```

By default Claude optimizes every instance in `data/`. To restrict a session to a subset, pass benchmark names and/or full IDs (`<name>_<instance>`) as positional arguments:

```bash
# Just one instance:
sbatch scripts/claude.sh matmul_2

# Every instance of one benchmark:
sbatch scripts/claude.sh matmul

# A mix of names and full IDs:
sbatch scripts/claude.sh matmul_2 conv_2d
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

Two helper scripts under [llm_transform/tools/](llm_transform/tools/) visualize the experiment logs.

Plot speedup over time, one curve per instance, for a single experiment:

```bash
python -m llm_transform.tools.plot_performance <EXPERIMENT_ID>
```

Compare multiple experiments side by side (one subplot per instance):

```bash
python -m llm_transform.tools.plot_performance_compare <EXPERIMENT_ID_1> <EXPERIMENT_ID_2> ...
```

Both scripts read from `logs/stats/` by default and save PNGs into the corresponding stats directory.

## 7. Running the validation tests

The validation harness checks that the equivalence verifier correctly flags illegal transform schedules. Test cases live in [tests/validation/](tests/validation/) — each file contains a kernel paired with a transform schedule that is either dependence-preserving or dependence-violating.

Run the full suite from the project root:

```bash
conda activate main
python test_mlir_validation.py
```

Useful flags:

- `-v` / `--verbose` — print captured stderr for each test.
- `--filter <substr>` — run only tests whose filename matches the substring (e.g. `--filter tiling`).

The harness requires the equivalence verifier shared libraries from [step 2](#2-building-the-equivalence-verifier).

## 8. Manual single-run execution

If you want to evaluate a single configuration outside of a Claude session, use [scripts/execute.sh](scripts/execute.sh). It runs the optimized configuration and the PyTorch reference, then prints the speedup:

```bash
sbatch scripts/execute.sh -i matmul_2 \
    -t resources/base_schedule.mlir \
    -p resources/base_passes.txt
```

Pass `--id <name>_<instance>` (or `-i`) plus any flags accepted by `llm_transform/utils/execution.py` (transform schedule path, MLIR passes file, LLVM passes/flags, etc.). The base no-op schedule lives at [resources/base_schedule.mlir](resources/base_schedule.mlir) and the default lowering pipeline at [resources/base_passes.txt](resources/base_passes.txt).

## 9. Project layout

A condensed view (full layout in [resources/context.md](resources/context.md)):

```txt
data/                     # Benchmarks: <name>/<instance>.mlir + sizes.json
resources/
  base_schedule.mlir      # No-op transform schedule (starting point)
  base_passes.txt         # Default MLIR lowering pipeline
  conda/                  # Conda environment definitions
llm_transform/            # Python package (installed via `pip install -e .`)
  mcp_server.py           # MCP tools: run_schedule, lower_schedule
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
