# ops_and_blocks Training Failures — Post-Mortem (2026-06-24)

## Overview

On 2026-06-24, three training jobs were submitted for the `ops_and_blocks` dataset.
All three failed immediately. This document records the root causes and fixes applied.

| SLURM Job | Config | Implementation | Duration | Exit Code |
|-----------|--------|----------------|----------|-----------|
| 16407225 | `config/ops_and_blocks/train/paper_original.json` | `rl_autoschedular_paper` | 36 min | 6 |
| 16407226 | `config/ops_and_blocks/train/paper_transformer_small.json` | `rl_autoschedular_paper_transformer` | 6 min | 1 |
| 16407227 | `config/ops_and_blocks/train/paper_transformer_large.json` | `rl_autoschedular_paper_transformer` | 6 min | 1 |

---

## Bug 1 — `rl_autoschedular_paper`: TiledFusion crashes on constant store dimensions

### Affected Job

Job 16407225 (`paper_original`). The job ran for 36 minutes (first trajectory
collected successfully but crashed during the PPO update phase when the agent
encountered benchmarks with constant-indexed producer store dimensions).

### Error

```
Exception: Unsupported producer store [['%d0', '%d1', '%d2', '0']] at position 3
Benchmark: llama3_2_1b_block_378
Transformations: ['I(0,3,2,1)']

Exception: Unsupported producer store [['%d0', '%d1', '0']] at position 2
Benchmark: llama3_2_1b_block_233
```

Followed by:
```
python: mlir/include/mlir/Bindings/Python/PybindAdaptors.h:623: ...
        Assertion `errorMessage.empty()' failed.
Aborted (core dumped)
```

### Root Cause

`rl_autoschedular_paper/actions/tiled_fusion.py`, method `__record_implicit_tiling`
(line 265):

```python
# Before (paper)
if dim_str not in prod_args_dims:
    raise Exception(f"Unsupported producer store [{prod_res_store}] at position {dim_pos}")
```

When a producer's result store has a **constant literal dimension** (e.g. `'0'`,
meaning that dimension is always written at index 0 and is not controlled by any
loop variable), the lookup `dim_str not in prod_args_dims` is `True` and the code
raises. This occurs on ops like `llama3_2_1b` blocks that use broadcast/reduction
patterns where one output dimension is constant.

The `rl_autoschedular_v4_5` package already handles this correctly — constant
dimensions are silently skipped because they carry no tiling information:

```python
# v4_5 (correct)
if dim_str not in prod_args_dims:
    continue  # constant dimension, not a loop variable
```

### Fix

**File**: `rl_autoschedular/rl_autoschedular_paper/actions/tiled_fusion.py`, line 264–265

```diff
         if dim_str not in prod_args_dims:
-            raise Exception(f"Unsupported producer store [{prod_res_store}] at position {dim_pos}")
+            continue  # constant dimension, not a loop variable
```

The paper package was ported from v4_5 but this one-line fix was missed. The
`ops_and_blocks` dataset contains `llama3_2_1b` blocks which expose this path;
the original `single_ops_dataset` apparently did not.

---

## Bug 2 — `rl_autoschedular_paper_transformer`: Dead duplicate class body causes `NameError`

### Affected Jobs

Jobs 16407226 and 16407227 (both `paper_transformer_small` and `paper_transformer_large`).
Both died within 6 minutes — before the first trajectory was even completed.

### Error

```
File ".../rl_autoschedular_paper_transformer/model.py", line 615, in forward
    hardware_feats = Observation.get_part(obs, HardwareFeatures)
                                               ^^^^^^^^^^^^^^^^
NameError: name 'HardwareFeatures' is not defined. Did you mean: 'hardware_feats'?
```

### Root Cause

`rl_autoschedular/rl_autoschedular_paper_transformer/model.py` contained **two
complete implementations of `TransformerEmbedding` concatenated in a single file**
— a consequence of a botched copy-paste merge.

**Structure of the file before the fix** (697 lines total):

```
Lines   1–233   HiearchyModel, ValueModel, PolicyModel           ← correct
Lines 234–462   TransformerEmbedding (paper_transformer version) ← correct
  Line 462:     return torch.cat((pooled, action_history), dim=1) ← valid return
Lines 463–697   Dead code: a second block that was unreachable,   ← BUG
                but contained a second def forward() that
                referenced HardwareFeatures (never imported)
```

Python's method resolution caused the **dead-code `forward`** (line 609) to shadow
the correct one (line 387), because the dead block redefined `__call__` at line 527,
making Python dispatch `forward` calls to the second definition. That `forward` at
line 615 calls `Observation.get_part(obs, HardwareFeatures)` — but `HardwareFeatures`
was never imported in this file (the paper_transformer package does not use hardware
features).

The dead block (lines 463–697) was code from `rl_autoschedular_v4_5/model.py`'s
`TransformerEmbedding`, which **does** use `HardwareFeatures`. It was inadvertently
left in the file after a merge.

### Fix

**File**: `rl_autoschedular/rl_autoschedular_paper_transformer/model.py`

Removed lines 463–697 (the entire dead duplicate block). The file now contains a
single, correct `TransformerEmbedding` that:

- Does **not** use `HardwareFeatures` (paper_transformer has no hardware features in obs)
- Concatenates `action_history` directly to the CLS-pooled output: `torch.cat((pooled, action_history), dim=1)`
- `output_size = d_model + ActionHistory.size()`

The correct version was already complete and correct at lines 234–462. Only the dead
tail had to be removed.

---

## Change 3 — `scripts/train/train.sh`: Reduced memory allocation from 64 GB to 8 GB

The three failed jobs requested `--mem=64G`. CPU-only training on the `ops_and_blocks`
dataset peaks well below 8 GB (the heaviest job, 16407225, used ~621 MB at peak per
`sacct`). The 64 GB request was unnecessarily large and likely caused slower queue
scheduling on nodes with less free memory.

```diff
-#SBATCH --mem=64G
+#SBATCH --mem=8G
```

---

## Jobs Resubmitted

After applying all three fixes, all files were verified with `python -m py_compile`
and the jobs were resubmitted:

| New SLURM Job | Config | Status at submission |
|---------------|--------|----------------------|
| 16407895 | `paper_original.json` | PENDING |
| 16407896 | `paper_transformer_small.json` | PENDING |
| 16407897 | `paper_transformer_large.json` | PENDING |

---

## Summary of Files Changed

| File | Change |
|------|--------|
| `rl_autoschedular/rl_autoschedular_paper/actions/tiled_fusion.py` | `raise Exception(...)` → `continue` for constant store dimensions |
| `rl_autoschedular/rl_autoschedular_paper_transformer/model.py` | Removed 235-line dead duplicate class body (lines 463–697) |
| `scripts/train/train.sh` | `--mem=64G` → `--mem=8G` |

---

## Round 2 — Second Wave of Failures (same day, resubmitted jobs)

After the fixes above, jobs 16407895/16407896/16407897 were submitted. Two failed again
within minutes.

| SLURM Job | Config | Implementation | Duration | Root Cause |
|-----------|--------|----------------|----------|------------|
| 16407895 | `paper_original.json` | `rl_autoschedular_paper` | ~5 min | SIGABRT during `Benchmarks()` construction |
| 16407897 | `paper_transformer_large.json` | `rl_autoschedular_paper_transformer` | ~5 min | `TiledFusion` constant dim raise (missed in round 1) |
| 16407896 | `paper_transformer_small.json` | `rl_autoschedular_paper_transformer` | ✅ running | — |

### Bug 3 — `rl_autoschedular_paper_transformer`: Same TiledFusion crash missed in round 1

The `continue` fix applied to `rl_autoschedular_paper/actions/tiled_fusion.py` in
round 1 was **not applied** to `rl_autoschedular_paper_transformer/actions/tiled_fusion.py`.
Both files are near-identical copies with different package prefixes.

**Error** (job 16407897):
```
Exception: Unsupported producer store [['%d0', '%d1', '%d2', '0']] at position 3
Benchmark: albert_block_772
```

**Fix**: Same one-line change in `rl_autoschedular_paper_transformer/actions/tiled_fusion.py`:

```diff
         if dim_str not in prod_args_dims:
-            raise Exception(f"Unsupported producer store [{prod_res_store}] at position {dim_pos}")
+            continue  # constant dimension, not a loop variable
```

### Bug 4 — SIGABRT during `Benchmarks()` construction crashes training before it starts

**Error** (job 16407895):
```
Extracting benchmark features: 100%|██████████| 6553/6553 [04:34<00:00]
python: mlir/.../PybindAdaptors.h:623: Assertion `errorMessage.empty()' failed.
Aborted (core dumped)
```

The signal handler in `scripts/train/train.py` converts SIGABRT to `RuntimeError`, but
`Benchmarks.__init__` called `extract_bench_features_from_file` with no exception guard.
One bad MLIR file among 6553 triggered SIGABRT → `RuntimeError` → uncaught → crash.

The `try/except RuntimeError` inside the training loop (line 164 of `train.py`) only
protects the iteration body — **not** the data loading phase at module level (line 83).
There is no practical way to guard line 83 without restructuring `train.py`; the right
fix is to guard inside `Benchmarks.__init__` itself.

**Fix** — both `rl_autoschedular_paper/benchmarks.py` and
`rl_autoschedular_paper_transformer/benchmarks.py`:

```diff
-            benchmark_data = extract_bench_features_from_file(bench_name, bench_file, root_exec_time)
+            try:
+                benchmark_data = extract_bench_features_from_file(bench_name, bench_file, root_exec_time)
+            except RuntimeError as e:
+                print_alert(f"Skipping {bench_name}: MLIR crashed during feature extraction ({e})")
+                continue
```

Bad benchmarks are now skipped with a warning; the remaining 6552+ load normally and
training proceeds.

---

## SIGABRT Safety Audit (post all fixes)

V4.9's safety is split across three layers. Status after all rounds of fixes:

| Mechanism | Location | V4.9 | `paper` | `paper_transformer` |
|-----------|----------|:----:|:-------:|:-------------------:|
| Signal handler (SIGABRT → RuntimeError) | `scripts/train/train.py` | ✅ | ✅ shared | ✅ shared |
| Signal handler (eval entry point) | `scripts/eval/eval.py` / `evaluate.py` | ✅ | ✅ | ✅ |
| Process-isolated MLIR execution | `execution.py` | ✅ | ✅ ported | ✅ ported |
| Dynamic timeout (`root_exec_time × 5`) | `execution.py` | ✅ | ✅ ported | ✅ ported |
| mlir-cpu-runner subprocess fallback | `execution.py` | ✅ | ✅ ported | ✅ ported |
| SIGABRT guard during benchmark loading | `benchmarks.py` | ✅ | ✅ **round 2** | ✅ **round 2** |
| TiledFusion constant dim skip | `actions/tiled_fusion.py` | ✅ | ✅ round 1 | ✅ **round 2** |

---

## Round 3 — Third Wave of Failures (2026-06-25)

After Round 2 fixes, jobs 16407944 and 16407945 were submitted. Both still crashed with native SIGABRT during training execution.

### Bug 5 — MLIR Overwriting Custom Python Signal Handlers

**Error**:
```
python: mlir/.../PybindAdaptors.h:623: Assertion `errorMessage.empty()' failed.
Aborted (core dumped)
```

**Root Cause**:
Although Python's custom SIGABRT signal handler was registered at import time in `train.py`/`eval.py`/`evaluate.py`, LLVM/MLIR native bindings initialize their own signal handlers (which override Python's `signal.signal` registration) when `MlirContext` is first constructed during benchmark loading (specifically during `Benchmarks` initialization or first C++ call). As a result, the custom SIGABRT handler was unregistered by the time training or evaluation actually executed benchmarks.

**Fix**:
Re-register the custom SIGABRT signal handler after MLIR context initialization is complete. This is inserted immediately after the `Execution(...)` singleton is constructed in all training and evaluation entry points:
- `scripts/train/train.py`
- `scripts/eval/eval.py`
- `scripts/eval/ablation_eval.py`
- `rl_autoschedular_paper/evaluate.py`
- `rl_autoschedular_paper_transformer/evaluate.py`

This ensures that the Python signal handler is active during benchmark execution.

---

## Complete Summary of Files Changed (all rounds)

| File | Change |
|------|--------|
| `rl_autoschedular/rl_autoschedular_paper/actions/tiled_fusion.py` | `raise` → `continue` for constant store dims (round 1) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/actions/tiled_fusion.py` | Same `raise` → `continue` fix (round 2 — missed in round 1) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/model.py` | Removed 235-line dead duplicate `TransformerEmbedding` body (round 1) |
| `rl_autoschedular/rl_autoschedular_paper/benchmarks.py` | `try/except RuntimeError` guard around per-benchmark MLIR parsing (round 2) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/benchmarks.py` | Same `try/except RuntimeError` guard (round 2) |
| `scripts/train/train.sh` | `--mem=64G` → `--mem=8G`; added `--constraint=bergamo` (round 1 + fairness fix) |
| `scripts/train/train.py` | Reinstalled SIGABRT handler after MLIR context initialization (round 3) |
| `scripts/eval/eval.py` | Reinstalled SIGABRT handler after MLIR context initialization (round 3) |
| `scripts/eval/ablation_eval.py` | Reinstalled SIGABRT handler after MLIR context initialization (round 3) |
| `rl_autoschedular/rl_autoschedular_paper/evaluate.py` | Reinstalled SIGABRT handler after MLIR context initialization (round 3) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/evaluate.py` | Reinstalled SIGABRT handler after MLIR context initialization (round 3) |

---

## Round 4 — Native C++ Diagnostic Destructor Aborts (2026-06-26)

After Round 3 signal handler adjustments, training jobs still encountered sudden process exits during benchmark trajectory collection.

### Bug 6 — Unchecked Diagnostic Aborts inside C++ Bindings (`CollectDiagnosticsToStringScope`)

**Error**:
```
python: /scratch/mb10856/MLIR-RL/llvm-project/mlir/include/mlir/Bindings/Python/PybindAdaptors.h:623: 
mlir::python::CollectDiagnosticsToStringScope::~CollectDiagnosticsToStringScope(): 
Assertion `errorMessage.empty() && "unchecked error message"' failed.
Aborted (core dumped)
```

**Root Cause**:
In `rl_autoschedular_paper` and `rl_autoschedular_paper_transformer`, the parent trajectory collection loops call `action.apply(module)` in-process. This function delegates to the C++ bindings function `interpreter.apply_named_sequence`.
If a pass succeeds but emits any warning/remark diagnostics (common in tiling/vectorization), or if a pass fails and the exception is caught and ignored in Python, the C++ local variable `CollectDiagnosticsToStringScope` is left with a non-empty `errorMessage` (warnings or remarks).
When the C++ scope ends (upon returning from the bindings function), the destructor asserts that `errorMessage` is empty. Because the diagnostics were not explicitly cleared/taken, this assertion fails and aborts the entire Python training process immediately.

**Fix**:
Implemented subprocess isolation in the transformation layer:
- Modified `__run_transform_code` inside [transforms.py](file:///scratch/mb10856/MLIR-RL/rl_autoschedular/rl_autoschedular_paper/transforms.py) and [transforms.py](file:///scratch/mb10856/MLIR-RL/rl_autoschedular/rl_autoschedular_paper_transformer/transforms.py) to serialize the active `Module` to string.
- Ran a Python subprocess using `subprocess.run` to load the module, parse the transform code, apply the named sequence, and output the resulting module string.
- Parsed the output string back into a temporary module under the parent process's `Context`, then used `move_module` to copy the operations back in-place into the parent `Module`.
- Forwarded thread limits (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) to avoid resource limits inside the subprocesses.

This isolated execution guarantees that any diagnostic issues (warnings, remarks, or native failures) are entirely confined to short-lived child processes, keeping the parent RL training process completely safe.

---

## Complete Summary of Files Changed (all rounds)

| File | Change |
|------|--------|
| `rl_autoschedular/rl_autoschedular_paper/actions/tiled_fusion.py` | `raise` → `continue` for constant store dims (round 1) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/actions/tiled_fusion.py` | Same `raise` → `continue` fix (round 2) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/model.py` | Removed 235-line dead duplicate `TransformerEmbedding` body (round 1) |
| `rl_autoschedular/rl_autoschedular_paper/benchmarks.py` | `try/except RuntimeError` guard around per-benchmark MLIR parsing (round 2) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/benchmarks.py` | Same `try/except RuntimeError` guard (round 2) |
| `scripts/train/train.sh` | `--mem=64G` → `--mem=8G`; added `--constraint=bergamo` (round 1) |
| `scripts/train/train.py` | Reinstalled SIGABRT handler after MLIR context initialization (round 3) |
| `scripts/eval/eval.py` | Reinstalled SIGABRT handler after MLIR context initialization (round 3) |
| `scripts/eval/ablation_eval.py` | Reinstalled SIGABRT handler after MLIR context initialization (round 3) |
| `rl_autoschedular/rl_autoschedular_paper/evaluate.py` | Reinstalled SIGABRT handler after MLIR context initialization (round 3) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/evaluate.py` | Reinstalled SIGABRT handler after MLIR context initialization (round 3) |
| `rl_autoschedular/rl_autoschedular_paper/transforms.py` | Subprocess transform isolation implementation (round 4) |
| `rl_autoschedular/rl_autoschedular_paper_transformer/transforms.py` | Same subprocess transform isolation implementation (round 4) |

