# Paper Mohammed Eval Pipeline — SIGABRT Safety Analysis

## Problem Statement

On 2026-06-22, evaluating `model_11000.pt` from `results/ops_and_blocks_results/paper_mohammed/` crashed after processing 728 of 1687 benchmarks. The eval job (SLURM ID 16378069) completed in 21 minutes but produced **no speedup results** — the process was killed by an MLIR SIGABRT during execution.

**Root cause**: `rl_autoschedular_paper`'s execution module runs MLIR in-process with no signal handling, no process isolation, and no fallback. V4.9's pipeline (`rl_autoschedular_v4_5` + `scripts/eval/eval.py`) has all three.

---

## Pipeline Comparison

### Entry Points

| Aspect | V4.9 (`scripts/eval/eval.py`) | paper_mohammed (`rl_autoschedular_paper/evaluate.py`) |
|--------|-------------------------------|-------------------------------------------------------|
| SIGABRT handler | ✅ Lines 14–16: `signal.signal(signal.SIGABRT, _sigabrt_handler)` | ❌ Absent — no `signal` import |
| Module selection | Dynamic via `get_autoschedular_impl()` → `import_autoschedular_module()` | Hardcoded: `from rl_autoschedular_paper.execution import Execution` |
| Checkpoint discovery | Sophisticated: `EVAL_DIR` env → auto-derive from `results_dir` + impl → `run_N/models/` | Simple: reads `models/` relative to CWD |
| Eval resumption | ✅ Tracks completed checkpoints in `_eval_checkpoint.txt` with `EVAL_FORCE`/`EVAL_STRIDE`/`EVAL_START`/`EVAL_END` | ❌ None |
| Logging | Slurm debug log via `logging.basicConfig` | No debug logging |

**Key finding**: `scripts/eval/eval.py` is the shared eval entry point for V4.9 (and all v4_5-based implementations). It installs the SIGABRT handler at import time *before* any MLIR code is touched. The paper has its own standalone `evaluate.py` that bypasses this entirely.

### Execution Modules

| Safety Mechanism | V4.9 (`rl_autoschedular/rl_autoschedular_v4_5/execution.py`) | paper_mohammed (`rl_autoschedular_paper/execution.py`) |
|---|---|---|
| SIGABRT handler | Installed in `scripts/eval/eval.py` at top-level | ❌ Missing entirely |
| Process isolation | ✅ `multiprocessing.Process` + `multiprocessing.Manager` inside `__execute_bufferized_code` (line 279) — runs MLIR in a **fresh child process** | Uses `BindingsProcess.call()` which uses `multiprocessing.get_context('fork')` — forks the MLIR state |
| BindingsProcess enabled? | Default `ENABLED = False` in `utils/bindings_process.py` — so `BindingsProcess.call()` is a **no-op passthrough** | Enabled via env `ENABLE_BINDINGS_PROCESS=1` in `rl_autoschedular_paper/utils/bindings_process.py` |
| Actual isolation strategy | When BindingsProcess is disabled, `__execute_bufferized_code` creates its own `multiprocessing.Process` with `Manager().dict()` for result passing — clean-process isolation | When BindingsProcess is enabled, it forks. When disabled, MLIR runs **in-process** with zero isolation |
| Timeout handling | ✅ Dynamic profiling-based timeout: `min(300, max(MIN_EXEC_TIMEOUT, root_exec_time * 5 / 1e9))` seconds. Checks `process.is_alive()` → `process.terminate()` | `BindingsProcess` has `timeout` param but `ENABLE_TIMEOUT = False` — **timeout is disabled by default** |
| Fallback execution | ✅ `__execute_code_with_cmd`: if bindings fail, falls back to `mlir-opt \| mlir-cpu-runner` as subprocess | ❌ No fallback — if `__execute_bufferized_code` fails, execution fails |
| SKIP_MLIR_BINDINGS | ✅ Returns immediately, triggers cmd fallback | ❌ Not supported |
| Execute return signature | `tuple[int, bool, bool, Optional[str]]` — includes error message | `tuple[int, bool, bool]` — no error message |

### Environment Modules

| Aspect | V4.9 `env.py` | paper_mohammed `env.py` |
|---|---|---|
| Shaped reward | ✅ `action.extras['shaped_reward'] = self.__shaped_reward(...)` | ❌ Not present |
| Success-contingent reward negation | ✅ Zeroes all intermediate rewards if final execution fails | ❌ Not present |
| Slowdown penalty | ✅ Zeros intermediate rewards if speedup < 1.0 | ❌ Not present |
| `apply_and_run_sequence` execution call | `Execution().execute_code(transformed_code, bench_name, seq, root_exec_time=...)` | `Execution().execute_code(transformed_module, bench_name, seq)` |
| Multi-run evaluation | ✅ `cfg.eval_runs` with aggregation (`min`/`median`/`mean`) | ❌ Single run only |
| Error message propagation | `run_time, run_ok, run_miss, run_err` — error messages flow up | `new_exec_time, exec_succeeded, cache_miss` — no error detail |
| Failed speedup value | `0.0` (division by None) | `1.0` (division by None gives 1.0 via guard) |

---

## Safety Mechanisms Summary

| # | Safety Mechanism | V4.9 | paper_mohammed | Impact |
|---|---|---|---|---|
| 1 | **SIGABRT signal handler** | ✅ `scripts/eval/eval.py:14–16` | ❌ Absent | A single MLIR SIGABRT **kills the entire eval job** |
| 2 | **Process-isolated MLIR execution** | ✅ `multiprocessing.Process` + `Manager` | ❌ In-process (unless `ENABLE_BINDINGS_PROCESS=1`) | SIGABRT in in-process MLIR = unrecoverable crash |
| 3 | **Profiling-based dynamic timeout** | ✅ `min(300, root_exec_time * 5)` | ❌ Hard-coded 600s via `BindingsProcess` | Hung MLIR can block for 10 minutes |
| 4 | **mlir-cpu-runner fallback** | ✅ `__execute_code_with_cmd` | ❌ No fallback | Bindings crash = total failure, no alternative path |
| 5 | **SKIP_MLIR_BINDINGS escape hatch** | ✅ | ❌ | No way to bypass broken bindings |
| 6 | **Eval resumption** | ✅ `_eval_checkpoint.txt` | ❌ | Restarting after crash re-evaluates everything |
| 7 | **fcntl file locking** | ✅ | ❌ (atomic replace only) | Race condition in concurrent cache reads |
| 8 | **Multi-run eval aggregation** | ✅ `eval_runs` + `eval_aggregation` | ❌ Single run | No statistical robustness |

---

## Crash Details (2026-06-22)

- **Job**: `mlir-eval` (SLURM ID 16378069), completed in 21 minutes
- **Checkpoint**: `model_11000.pt`
- **Benchmarks loaded**: 1687
- **Benchmarks processed before crash**: 728
- **Last benchmark processed**: `llama3_2_1b_block_1170`
- **Error**: `Aborted (core dumped)` — MLIR SIGABRT
- **Results written**: Only `run_0/logs/eval/entropy` (39K entries from inference). No speedup, reward, or exec time data.
- **Slurm exit code**: 0 (Slurm thinks it succeeded — the SIGABRT killed the Python process cleanly from Slurm's perspective)

---

## Implemented Fixes (2026-06-22)

Options 1 and 2 were implemented together. All changes verified with `py_compile`.

### Files Modified

| File | Change |
|------|--------|
| `rl_autoschedular_paper/evaluate.py` | Added SIGABRT signal handler |
| `rl_autoschedular_paper/execution.py` | Ported process-isolated execution, dynamic timeout, mlir-cpu-runner fallback |
| `rl_autoschedular_paper/env.py` | Updated `execute_code` call to unpack 4-tuple, pass `root_exec_time` |
| `rl_autoschedular_paper/baseline.py` | Updated `execute_code` call to unpack 4-tuple |

### Detail of Changes

#### 1. SIGABRT Handler (`evaluate.py`)

```python
import signal

def _sigabrt_handler(signum, frame):
    raise RuntimeError("MLIR native code crashed (SIGABRT) — caught and continuing")
signal.signal(signal.SIGABRT, _sigabrt_handler)
```

Registered at module import time (before any MLIR code is touched). Converts native SIGABRT into a catchable Python exception. If a benchmark triggers SIGABRT during inference or execution, the eval loop catches it and continues to the next benchmark instead of dying.

#### 2. Process-Isolated Execution (`execution.py`)

Replaced the in-process `__execute_bufferized_code` (which ran MLIR via `execution_engine.invoke()` directly in the parent process) with:

- **`__execute_bufferized_code_isolated`**: Spawns a `multiprocessing.Process` + `Manager().dict()`. The worker process creates its own `Context()`, parses the serialized MLIR code, runs the pass pipeline, and executes via `ExecutionEngine`. The parent joins with timeout. If the child dies (SIGABRT, segfault, etc.) the parent survives and gets `result_dict.get('success') == False`.

- **`__execute_bufferized_code_wrapper`**: Orchestrates isolated execution → fallback. Serializes the `Module` to string (`str(module)`) for cross-process transfer, then calls the isolated worker. On failure, falls through to `mlir-cpu-runner`.

#### 3. Dynamic Timeout (`execution.py`)

```python
min_timeout = int(os.environ.get("MIN_EXEC_TIMEOUT", "300"))
timeout_s = 300
if root_exec_time and root_exec_time > 0:
    timeout_s = min(300, max(min_timeout, int((root_exec_time / 1e9) * 5)))
```

Mirrors V4.9's profiling-based safeguard. Default 300s; if `root_exec_time` is known, allows 5x slowdown margin with a floor of `MIN_EXEC_TIMEOUT`. Hung MLIR processes are terminated after timeout instead of blocking indefinitely.

#### 4. mlir-cpu-runner Fallback (`execution.py`)

```python
def __execute_code_with_cmd(self, code_str: str, timeout_s: int) -> tuple[int, bool]:
```

If the Python bindings crash or timeout, falls back to a subprocess pipeline: `mlir-opt ... <file> | mlir-cpu-runner ...`. This provides a second execution path that is fully process-isolated and doesn't depend on MLIR Python bindings stability.

#### 5. Updated API Signatures

`execute_code` now returns a 4-tuple `(exec_time, success, cache_miss, error_msg)` and accepts an optional `root_exec_time` parameter:

```python
def execute_code(self, module, bench_name, seq, root_exec_time=None) -> tuple[int, bool, bool, Optional[str]]:
```

All callers updated accordingly:
- `env.py:125` — unpacks 4 values, passes `root_exec_time=self.benchmark_data.root_exec_time`
- `baseline.py:43` — unpacks 4 values (discards extras)

### Safety Mechanisms After Fix

| # | Safety Mechanism | Before | After |
|---|---|---|---|
| 1 | SIGABRT signal handler | ❌ | ✅ |
| 2 | Process-isolated MLIR execution | ❌ (in-process) | ✅ (`multiprocessing.Process`) |
| 3 | Dynamic timeout | ❌ (hardcoded 600s) | ✅ (profiling-based, `root_exec_time * 5`) |
| 4 | mlir-cpu-runner fallback | ❌ | ✅ |
| 5 | Error message propagation | ❌ | ✅ (4th return value) |

### To Re-run the Eval

```bash
sbatch --cpus-per-task=12 --mem=16G --time=04:00:00 \
  scripts/eval/eval.sh config/paper_mohammed/paper_eval.json --checkpoint 11000
```

### Remaining Gaps (Not Ported)

| Mechanism | Status | Notes |
|-----------|--------|-------|
| Eval resumption (`_eval_checkpoint.txt`) | ❌ Not ported | Restarting after crash re-evaluates everything. Low priority for paper artifact. |
| `fcntl` file locking | ❌ Not ported | Atomic replace (`os.replace`) is used instead. Sufficient for single-process eval. |
| Multi-run eval aggregation | ❌ Not ported | Paper uses single-run eval by design. |
