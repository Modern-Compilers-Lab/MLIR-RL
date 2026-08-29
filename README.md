# MLIR-RL — Learning to Optimize MLIR Code

Research programme of the [Modern Compilers Lab](https://github.com/Modern-Compilers-Lab) on
**automatic code optimization for the MLIR compiler**, driven by reinforcement learning and, more
recently, by LLM agents.

This is the *monorepo* holding four related research directions. If you are new here, read
[§1](#1-what-the-research-is-about) and [§2](#2-the-research-arc), then jump to the component you
have been assigned via the [index](#8-index--where-to-go-next).

---

## Table of contents

1. [What the research is about](#1-what-the-research-is-about)
2. [The research arc](#2-the-research-arc)
3. [Prerequisites — which MLIR do I use?](#3-prerequisites--which-mlir-do-i-use)
4. [Conda environments](#4-conda-environments)
5. [Cluster (SLURM) conventions](#5-cluster-slurm-conventions)
6. [Target hardware](#6-target-hardware)
7. [Secrets and configuration files](#7-secrets-and-configuration-files)
8. [Index — where to go next](#8-index--where-to-go-next)

---

## 1. What the research is about

A compiler like MLIR can express the *same* computation in a huge number of semantically equivalent
ways. For dense loop nests — matrix multiplication, convolution, pooling, elementwise operations —
the choice among those forms is worth orders of magnitude of runtime. The choice is called a
**schedule**: an ordered sequence of loop transformations, each with parameters.

```
        linalg kernel                    schedule (what we search for)
   ┌────────────────────┐        ┌───────────────────────────────────────┐
   │ linalg.matmul      │        │ 1. parallelize   (num_threads=28)     │
   │   ins(%A, %B)      │  ───►  │ 2. tile          (tile_sizes=[8,8,0]) │  ───► fast code
   │   outs(%C)         │        │ 3. tile          (tile_sizes=[0,0,32])│
   │ {tag="operation_0"}│        │ 4. vectorize     ([4,4,4])            │
   └────────────────────┘        └───────────────────────────────────────┘
```

Concretely, throughout this repository:

- **Kernels** are MLIR `linalg`-dialect functions. The single operation to optimize is marked with
  `attributes {tag = "operation_0"}` so every tool can find it unambiguously.
- **Transformations** are expressed in the MLIR [**Transform dialect**](https://mlir.llvm.org/docs/Dialects/Transform/)
  — tiling, loop interchange, vectorization, parallelization, packing/promotion, im2col.
- **Measurement is real.** Every candidate schedule is applied, bufferized, lowered through a fixed
  LLVM pass pipeline, compiled, and *executed on a reserved compute node*. There is no cost model.
- **The objective** is speedup, measured either over the unoptimized MLIR baseline or over PyTorch
  running the identical shape on the identical hardware.

Casting this as reinforcement learning gives: **state** = an encoding of the loop nest plus the
transformations applied so far; **action** = pick a transformation and its parameters; **reward** =
the measured speedup. The recurring research question across all four components below is *how to
define the action space* — which turns out to be the real bottleneck, not the policy.

## 2. The research arc

Four directions, in the order they were explored:

| # | Component | Core idea | Status |
|---|---|---|---|
| 1 | [rl_autoschedular/](rl_autoschedular/) | **Hand-designed** action space + custom hierarchical PPO. This is the **published system**. | Paper; reproduce via the [official artifact](https://github.com/mohph197/MLIR-RL-artifact) |
| 2 | [llm_action/](llm_action/) | LLM agents **synthesize the action space** (enumerate → implement → explore), then MaskablePPO trains on the generated actions. | Active |
| 3 | [llm_transform/](llm_transform/) | Skip RL entirely: a Claude Code agent **directly rewrites** transform schedules, MLIR passes, and LLVM flags per kernel. | Complete, self-documented |
| 4 | [iql/](iql/) | **Offline RL** (Implicit Q-Learning) over logged trajectories, reusing the phase-1 environment. | Prototype |

Why each phase happened:

- **Phase 1** built the full RL loop and showed it works. It also exposed the cost centre: the seven
  hand-written actions in [rl_autoschedular/actions/](rl_autoschedular/actions/) each encode legality
  rules, composition constraints, and a hand-tuned parameter discretization. Every new operator
  family or transformation is a fresh round of senior-compiler-engineer time.
- **Phase 2** attacks exactly that: can an LLM agent *write* the action space — implement each action
  in the Transform dialect, test it against real execution, and empirically discover how actions
  compose? See [llm_action/](llm_action/).
- **Phase 3** asks the sceptical question: if an LLM agent is good enough to write the actions, is
  the RL policy needed at all, or can the agent just optimize each kernel directly? See
  [llm_transform/](llm_transform/).
- **Phase 4** attacks sample efficiency. Every environment step costs a real compile + execute
  (seconds to minutes), so online RL is expensive. Offline RL reuses logged trajectories across many
  gradient updates. See [iql/](iql/).

Phases 1 and 4 share the code at the repository root (`train.py`, `evaluate.py`, `utils/`,
`rl_autoschedular/`, `tools/`). Phases 2 and 3 are self-contained subprojects with their own
environments and entry points.

## 3. Prerequisites — which MLIR do I use?

This is the question that costs newcomers the most time. **There are two routes, and which one you
need depends on the component.**

### Route A — conda-provided MLIR (no source build)

Used by [llm_action/](llm_action/) and [llm_transform/](llm_transform/). The conda-forge packages
`mlir-python-bindings` / `libmlir` ship everything needed:

- the `mlir` Python package (IR, `PassManager`, `ExecutionEngine`),
- the binaries `mlir-opt`, `mlir-translate`, `mlir-runner`, `llc`, `opt`, `clang`,
- the runtime shared libraries `lib/libmlir_c_runner_utils.so`, `lib/libmlir_runner_utils.so`,
  `lib/libomp.so` — these are what `MLIR_SHARED_LIBS` must point at.

Install with, e.g. (this is what [llm_transform/scripts/setup.sh](llm_transform/scripts/setup.sh) does):

```bash
conda install -y -c conda-forge python=3.11 clang=21.1.8 clangxx=21.1.8 lld=21.1.8 \
    llvm-openmp=21.1.8 mlir-python-bindings=21.1.8 cmake ninja
```

**Prefer this route.** It is faster, reproducible, and it is what the working `mlir` environment
uses (LLVM/MLIR 21).

### Route B — LLVM/MLIR built from source

Required only by [rl_autoschedular/](rl_autoschedular/), because the custom C++ tools in
[tools/](tools/) (`ast_dumper`, `vectorizer`, `pre_vec`) are `add_llvm_executable` targets — they
need a real LLVM build tree with `MLIRConfig.cmake` and `LLVMConfig.cmake`.

```bash
git clone --depth 1 -b release/19.x https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
  -DLLVM_TARGETS_TO_BUILD=X86 \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build --target check-mlir
```

Needs CMake ≥ 3.20, Ninja, GCC/G++ 13.2, LLD, Python ≥ 3.11. Budget an hour or more.

Then point the environment at it (see [rl_autoschedular/README.md](rl_autoschedular/README.md)):

```bash
export LLVM_BUILD_PATH=/path/to/llvm-project/build
export PATH="$LLVM_BUILD_PATH/bin:$PATH"
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PYTHONPATH"
export MLIR_SHARED_LIBS="$LLVM_BUILD_PATH/lib/libomp.so,$LLVM_BUILD_PATH/lib/libmlir_c_runner_utils.so,$LLVM_BUILD_PATH/lib/libmlir_runner_utils.so"
```

> The reference recipe for the *published* system is the artifact repository, which pins its
> versions and ships a Dockerfile: <https://github.com/mohph197/MLIR-RL-artifact>.

## 4. Conda environments

| Environment | Python | Used by | Contains |
|---|---|---|---|
| **`mlir`** | 3.14 | [llm_action/](llm_action/) — **verified working** | MLIR/LLVM 21 + Python bindings, `torch` 2.10 (cpu), `gymnasium`, `stable-baselines3` + `sb3_contrib`, `fastmcp`, `dask` + `dask_jobqueue`, `wandb`, `anthropic`, `agno`, `pydantic` |
| `llm_transform` | 3.11 | [llm_transform/](llm_transform/) | clang/clangxx/lld/llvm-openmp/mlir-python-bindings 21.1.8, poetry, torch (cpu), matplotlib, fastmcp |
| `llvm-build` | — | [rl_autoschedular/](rl_autoschedular/) (`CONDA_ENV` in the root `.env`) | source-built LLVM toolchain + the phase-1 Python deps |
| `torch-cpu` | 3.11 | PyTorch reference baselines | torch only |

Exported specifications live in [env/](env/):

```bash
conda env create -f env/mlir-env.yml        # the mlir environment
conda env create -f env/main-env.yml
conda env create -f env/torch-cpu-env.yml
```

The `llm_transform` environment is not exported here — it is created by its own idempotent setup
script, [llm_transform/scripts/setup.sh](llm_transform/scripts/setup.sh).

Python dependencies for the root (phase 1 / phase 4) code are in [requirements.txt](requirements.txt).

On the cluster, `conda` needs to be brought into the shell first — every SLURM script in this repo
starts with:

```bash
module load miniconda-nobashrc 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate <env>
```

## 5. Cluster (SLURM) conventions

All experiments run on SLURM. Three conventions are shared by every component.

### Partition, reservation, QoS

```bash
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --qos=c2
```

`--qos=c2` has `UsageFactor=0`, which is what lets jobs through the `AssocGrpCpuLimit` on the
`default` account. It is already set in
[llm_action/scripts/mlir.sh](llm_action/scripts/mlir.sh),
[torch.sh](llm_action/scripts/torch.sh),
[train.sh](llm_action/scripts/train.sh) and
[evaluate.sh](llm_action/scripts/evaluate.sh).
It is **not** set in the `llm_action/scripts/claude_*.sh` agent scripts — add it there if you hit
the CPU-limit error.

### Thread affinity — mandatory for valid timings

Every job that *measures* anything must export this block, reproduced verbatim from
[llm_action/scripts/mlir.sh](llm_action/scripts/mlir.sh):

```bash
export OMP_NUM_THREADS=$(nproc)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE=static
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=passive
export KMP_BLOCKTIME=0
```

Without it, OpenMP threads migrate between the node's two NUMA domains, caches thrash, and
measurements become both slow and noisy — silently. This is the single most common cause of
"my numbers don't reproduce".

### Log locations

| Component | Logs |
|---|---|
| `llm_action` agent runs | `llm_action/logs/jobs/claude/{enum,implem,explore,optim}_<jobid>.{out,err}` |
| `llm_action` training / evaluation | `llm_action/logs/jobs/{train,evaluate}/<jobid>.{out,err}` |
| `llm_action` one-off execution | `llm_action/logs/jobs/{mlir,torch}/<jobid>.{out,err}` |
| `llm_action` exploration transcripts | `llm_action/logs/mcp/v<x>/<kernel>_<datetime>.md` |
| `llm_transform` | `llm_transform/logs/claude/<jobid>.log`, `llm_transform/logs/stats/<EXPERIMENT_ID>/` |
| `rl_autoschedular` | `logs/<jobname>_<jobid>.debug`, `logs/train.{out,err}`, `logs/neptune/` |

## 6. Target hardware

Every heuristic, tile-size vocabulary, and vector-width constraint in this repository is derived from
one machine. Know it:

- **Intel Xeon E5-2680 v4** (Broadwell), **28 physical cores** = 2 sockets × 14, **2 NUMA nodes**,
  no SMT.
- **AVX2 + FMA, no AVX-512** → 256-bit registers = **8 × f32** or **4 × f64** lanes.
- **L1d ≈ 32 KB/core**, **L2 ≈ 256 KB/core**, shared **L3 ≈ 35 MB** per socket.

This is encoded as `N_CORES = 28` in [llm_action/src/config.py](llm_action/src/config.py) and stated
to the LLM agents in [llm_action/resources/prompts/v1/system_description.md](llm_action/resources/prompts/v1/system_description.md).
Consequences that show up everywhere:

- Tile-size vocabularies are `[0, 4, 8, 16, 32]` — chosen so a tile row fits in L1.
- Vectorization sizes must **divide** the loop bounds (otherwise a dynamic remainder loop appears
  that downstream lowering cannot vectorize) and stay under
  `VECTORIZATION_SIZE_LIMIT = 2048` elements.
- Parallelism sweet spot is ~256 blocks; both far fewer and far more are measurably worse.
- On a different machine (AVX-512, different cache sizes), the learned policies and the discovered
  schedule graphs do not transfer unchanged.

A quick-reference sheet with measured numbers lives in
[llm_action/docs/MLIR_OPTIMIZATION_REFERENCE.md](llm_action/docs/MLIR_OPTIMIZATION_REFERENCE.md).

## 7. Secrets and configuration files

`.env` files are **gitignored** ([.gitignore](.gitignore)) and hold credentials. Never commit them
and never paste their contents into documentation, issues, or agent prompts.

| File | Template | Holds |
|---|---|---|
| `.env` (repo root) | [.env.example](.env.example) | `NEPTUNE_PROJECT`, `NEPTUNE_TOKEN`, `LLVM_BUILD_PATH`, `MLIR_SHARED_LIBS`, `AST_DUMPER_BIN_PATH`, `VECTORIZER_BIN_PATH`, `PRE_VEC_BIN_PATH`, `CONDA_ENV`, `CONFIG_FILE_PATH` |
| `llm_action/.env` | [llm_action/.env.example](llm_action/.env.example) | `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `MLIR_SHARED_LIBS`, `AST_DUMPER_BIN_PATH` |
| `llm_transform/scripts/env.local.sh` | generated by `setup.sh` | conda env name, MLIR paths |

Experiment tracking differs per component: **Neptune** for `rl_autoschedular`
([neptune_sync.py](neptune_sync.py), [scripts/neptune-sync.sh](scripts/neptune-sync.sh)) and
**Weights & Biases** for `llm_action`.

## 8. Index — where to go next

### Component documentation

| Read this | If you are working on |
|---|---|
| [rl_autoschedular/README.md](rl_autoschedular/README.md) | The published RL autoscheduler (phase 1) |
| [llm_action/README.md](llm_action/README.md) | LLM-synthesized action spaces + PPO (phase 2) |
| [llm_transform/README.md](llm_transform/README.md) | Direct LLM optimization of MLIR (phase 3) |
| [iql/README.md](iql/README.md) | Offline RL (phase 4) |

### Deep references

- [llm_action/DOCUMENTATION.md](llm_action/DOCUMENTATION.md) — the full technical dossier for phase 2:
  the three-layer pipeline, the action contract, the MDP, the reward design, with figures. Read this
  after the `llm_action` README.
- [llm_action/docs/MCP.md](llm_action/docs/MCP.md) and
  [llm_action/docs/MCP_MINIMAL.md](llm_action/docs/MCP_MINIMAL.md) — the MCP tool catalogue that
  grounds the LLM agents in the compiler.
- [llm_action/docs/MLIR_OPTIMIZATION_REFERENCE.md](llm_action/docs/MLIR_OPTIMIZATION_REFERENCE.md) —
  hardware + transformation quick reference (what actually made kernels fast).
- [llm_action/docs/llm_pipeline_metrics.md](llm_action/docs/llm_pipeline_metrics.md) — measured token,
  time and tool-call cost of running the agent pipeline.
- [llm_transform/resources/context.md](llm_transform/resources/context.md) — full technical overview
  of the direct-optimization pipeline.

### Paper artifact

The evaluation artifact for the published MLIR-RL paper is a **separate repository**:

> **<https://github.com/mohph197/MLIR-RL-artifact>**

Use it — not this repository — when the goal is to *reproduce the paper's numbers*. It ships a
Dockerfile, pinned dependencies (Python 3.11, LLVM/MLIR from source, Clang 21.1.5), pre-trained model
checkpoints, and `scripts/train.sh` / `scripts/evaluate.sh` / `scripts/paper.sh`, the last of which
regenerates the paper's speedup tables and figures into `paper/results/` and `paper/figures/`. This
repository is the *research working copy*: it has since diverged and contains three further
directions the paper does not cover.
