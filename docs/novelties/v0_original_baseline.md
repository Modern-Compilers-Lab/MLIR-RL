# rl_autoschedular — Baseline Implementation Deep Dive

This document provides a comprehensive breakdown of the original `rl_autoschedular` package (the baseline that all later versions fork from).

---

## 1. State Representation

### BenchmarkFeatures
Holds everything for a single benchmark:
- `code`: raw MLIR string.
- `operation_tags`: list of op identifiers in DAG order.
- `operations`: dict mapping tag → **OperationFeatures**.
- `root_exec_time`: unoptimized execution time in nanoseconds.

### OperationFeatures (per op)
- `operation_name` / `operation_type`: `generic`, `matmul`, `conv`, `pooling`, `add`.
- `op_count`: arithmetic ops (`+`, `-`, `*`, `/`, `exp`).
- `load_data` / `store_data`: access matrices as string formulas.
- `nested_loops`: list of **NestedLoopFeatures** (`arg`, `lower_bound`, `upper_bound`, `step`, `iterator_type: parallel/reduction`).
- `producers` / `consumers`: dependency graph edges.
- `vectorizable`: bool flag.
- `pre_actions`: actions already applied (e.g., implicit tiling from fusion).

### OperationState (RL state)
- `operation_tag`: current op.
- `original_operation_features`: frozen initial features.
- `operation_features`: current mutable features.
- `producer_tag`, `producer_operand_idx`, `producer_features`: info needed for fusion.
- `transformation_history`: list of action lists per op step.
- `terminal`: episode done flag.

Feature extraction is done by an external C++ binary (`AST_DUMPER_BIN_PATH`) that parses the MLIR and dumps a structured AST.

---

## 2. Observation Space

Observation = concatenation of 5 parts:

| Part | Content | Size |
|---|---|---|
| **OpFeatures** | One-hot op type, loop upper bounds (log2 or /4096), iterator types (parallel=1), load/store access matrices (max_stores_loads × max_load_store_dim × max_num_loops flattened), arithmetic op counts | dynamic |
| **ProducerOpFeatures** | Same as OpFeatures for the producer; all zeros if none | same |
| **ActionHistory** | Flattened one-hot encoding of all past actions in the current operation sequence | dynamic |
| **NumLoops** | Scalar count of nested loops | 1 |
| **ActionMask** | Boolean mask over action type + all parameter spaces | dynamic |

Total size depends on config (`max_num_loops`, `max_num_stores_loads`, etc.). Built by `Observation.from_state()`.

---

## 3. Action Space

Six action types exist. Each defines `symbol`, `params_size()`, `network_output_size()`, `mask_size()`, `history_size()`.

### NoTransformation (NT)
- Terminal. Ends the current operation episode.
- No parameters.

### Tiling (T)
- Parameters: one per loop (`max_num_loops`). `0` = no tile. `> 0` maps to `2^(param-1)`.
- Mask: per-loop valid tile sizes (only divisors of the upper bound).
- Feature update: loop upper bounds replaced by tile sizes; producers cleared.

### TiledParallelization (TP)
- Inherits Tiling.
- Parallelizes non-reduction loops (tile size = loop bound), tiles reduction loops.
- Forbidden if any prior Tiling already exists on this op.

### TiledFusion (TPF)
- Inherits TP.
- Fuses producer into the consumer's `scf.forall` loop created by TP.
- `__new__` trick: if all `parallel_params` are `0`, falls back to plain TP (no fusion).
- Allowed only if `state.producer_tag` exists.
- **Complex side effects**: updates the benchmark-level producer/consumer graph, inserts a new fused producer op into `operation_tags`, removes or updates the original producer. Implicitly adds pre-tiling to the producer.
- `_apply_ready` runs `transform_TF` then `transform_tile`.

### Interchange (I)
- Reorders loops. Three modes configured via `interchange_mode`:
  - `enumerate`: pre-computed 1c/2c/3c swap candidates. Network outputs a candidate index.
  - `pointers`: picks loop indices one-by-one to build a permutation. Can be multi-step (`ready=False` until permutation is complete).
  - `continuous`: single scalar decoded via the Lehmer code / factorial number system into a full permutation. Uses a `Normal` distribution with learned `log_std`.
- Mask: hides invalid candidates, already-picked indices, or padding.
- Feature update: reorder `nested_loops`. If conv/pooling and not identity, `vectorizable=False`.

### Vectorization (V)
- Terminal.
- For conv/pooling: may first transpose (`nhwc_fhwc` → `nhwc_hwcf`), decompose via tiling, then pre-vectorize.
- Allowed only if iteration space ≤ `vect_size_limit`.
- Failures during preprocessing are ignored (returns original code).

---

## 4. Model (`HiearchyModel`)

### LSTMEmbedding
- MLP: `OpFeatures.size()` → 512 → 512. ELU + Dropout(0.225).
- LSTM: input 512, hidden 411. Consumer embeddings and producer embeddings fed as a 2-step sequence.
- Output: final LSTM hidden (411) concatenated with `ActionHistory`.

### PolicyModel
- Backbone: `LSTMEmbedding.output_size` → 512 → 512 → 512. ReLU.
- Heads:
  - Head 0: action selection logits (`ActionSpace.size()` = 6).
  - Heads 1..N: parameter logits for each action type (size = `action.network_output_size()`).
- `forward()` builds masked distributions via `ActionSpace.distributions()`. Invalid actions/parameters get `-inf`.

### ValueModel
- Same LSTMEmbedding → 512 → 512 → 512 → 1.
- Output: state value scalar.
- Loss: MSE vs returns. Optional clipping: `values + clamp(new_values - values, -0.2, 0.2)`.

### Sampling
- `model.sample(obs, greedy=False, eps=None)`.
- If `eps` is set: with probability `eps`, sample from a uniform distribution over allowed actions.
- Otherwise: sample from policy distributions.
- Returns `(actions_index, actions_log_p, entropies)`.

---

## 5. Reward Function

All step rewards are `0` during trajectory collection. Only the final execution yields a reward.

In `Env.__action_reward()`:
- Transformation failed (apply exception): **-5.0**
- Execution failed / wrong results: **-20.0**
- Success: `log10(old_exec_time / new_exec_time)`

Speedup = 1 → reward 0. Speedup > 1 → positive. Speedup < 1 → negative. Log scale compresses large speedups.

The reward is assigned to the **last action** in the sequence. Intermediate actions in the same op get `0.0`.

---

## 6. Environment Dynamics (`Env`)

- `reset(benchs, bench_idx)`: pick benchmark, create `OperationState` for the last operation (`operation_idx = -1`). Operations are scheduled in reverse DAG order (outermost / last op first).
- `step(state, action)`:
  - Copy state.
  - Update features via `action.update_features()`.
  - Check terminal: `action.terminal` or `action_failed` or `step_count == truncate`.
- `get_next_op_state(state)`: if current op is terminal and benchmark is not done, move to the previous op (`current_op_index - 1`). Carry over `transformation_history`.
- `apply_and_run_sequence(seq)`:
  - Apply actions **in reversed order** (innermost op first). Within each `op_seq`, apply actions sequentially.
  - If an action apply fails: punish (-5) and stop applying further actions in that op.
  - After all are applied, compile & execute the transformed MLIR.
  - Return list of rewards (mostly 0, last = final reward), speedup, exec_time, cache_miss.

---

## 7. Execution Engine (`Execution`)

Singleton.

- `execute_code(code, bench_name, seq)`:
  - Check cache by key = `'|'.join(str(actions) per op_seq)`.
  - Cache hit: return stored time.
  - Cache miss: run `transform_bufferize_and_lower_v()` then execute.
- Compilation pipeline (`__execute_bufferized_code`):
  - Parse MLIR module.
  - Run a massive `PassManager` pipeline: canonicalize, buffer-deallocation, linalg→loops, scf→OpenMP, expand-strided-metadata, memref→LLVM, vector→LLVM, math→LLVM, func→LLVM, etc.
  - Create `ExecutionEngine` with `opt_level=3`, shared libs from `MLIR_SHARED_LIBS`.
  - Create zero-filled input memrefs via Python bindings.
  - Invoke `main` twice (warmup + measure). Capture `delta` (int64) from the output struct.
  - Returns `(exec_time_ns, success)`.
- Caching: local JSON file (`exec_data_file`) + optional global `main_exec_data`. Updated after each trajectory.

---

## 8. Training Pipeline (PPO)

### Trajectory Collection (`collect_trajectory`)
- Sample `cfg.bench_count` benchmarks randomly.
- Initialize one `Env` + `OperationState` + `Observation` per benchmark.
- While active states exist:
  - Batch observations, run `model.sample()`.
  - For each active state: get action, `env.step()`, build next observation.
  - If terminal: `get_next_op_state()` if more ops remain, else `done=True`.
  - Record `(num_loops, action_index, obs, next_obs, log_p, 0.0, done)` in `TrajectoryCollector`.
- After all done: distributed execution via `DaskManager.map_states(__execute_states)`. Workers run `env.apply_and_run_sequence()` and return rewards.
- Fill rewards into collectors. Update execution cache.
- Concatenate all collectors → `TrajectoryData`.

### Trajectory Data Processing
`TrajectoryData.update_attributes(model)`:
- Compute `values` = `value_model(obs)`.
- Compute `next_values` = `value_model(next_obs)`.
- Compute `actions_old_log_p` = current policy log-prob at collected actions.
- Compute `off_policy_rates` (rho) = `exp(old_log_p - behav_log_p)`. `1.0` if no epsilon exploration and no replay.
- Compute **returns** with a V-trace-like correction:
  - `returns[t] = values[t] + (rewards[t] + gamma * last_return - values[t]) * min(rho, 1)`. Gamma = 1.0.
- Compute **GAE advantages**:
  - `delta = rewards + gamma * next_values - values`.
  - `advantages[t] = delta + gamma * lambda * last_advantage`. Lambda = 0.95.

### PPO Update (`ppo_update`)
- Create DataLoader from trajectory. Optional `TopKAdvantageSampler` (if `reuse_experience='topk'`).
- For `cfg.ppo_epochs`:
  - For each batch:
    - `actions_log_p, new_values, entropies = model(obs, actions_index)`.
    - Policy loss: clipped surrogate with off-policy rates.
      - `ratios = exp(clamp(log_p_new - log_p_behav, -80, 80))`.
      - `surr1 = ratios * advantages`.
      - `surr2 = clamp(ratios, (1-clip)*rho, (1+clip)*rho) * advantages`.
      - `loss = -min(surr1, surr2).mean()`.
    - If `value_epochs == 0`: add value loss + entropy loss to the same optimizer step.
    - Else: value trained separately.
    - Normalize advantages: `standard` (mean/std) or `max-abs`.
    - Gradient clip norm = 0.5.

### Value Update (`value_update`)
- Same batches, but only `value_model` forward + MSE loss.

### Evaluation (`evaluate_benchmarks`)
- Greedy sampling (`model.sample(..., greedy=True)`).
- Same distributed execution.
- Logs per-benchmark speedups, average speedup, entropy.

---

## 9. Key Design Patterns

- **Singletons**: `Config`, `DaskManager`, `FileLogger`, `Execution`. Must be initialized in the correct order.
- **Hierarchical action index**: `actions_index` tensor = `[action_type, param1, param2, ...]`. Cumulative sizes managed by `ActionSpace`.
- **Action masking**: invalid choices are masked to `-inf` before `Categorical` / `Normal`. Ensures the agent never samples illegal transformations.
- **Multi-step actions**: `Interchange` with `LevelsPointers` can have `ready=False`. The state blocks other actions until the permutation is complete.
- **Reverse op scheduling**: `operation_tags` are processed from last to first (index -1, -2, ...). This matches the MLIR DAG where consumers are often scheduled after producers.
- **Replay / experience reuse**: `old_trajectory` is concatenated with the new one. `TrajectoryData.__add__` truncates to `cfg.replay_count` recent trajectories. `TopKAdvantageSampler` focuses updates on the highest-advantage timesteps.

---

## 10. Entrypoint Flow

`scripts/train.py`:
1. Load `.env`, config, benchmarks, initialize singletons.
2. Initialize `HierarchyModel`, Adam optimizer.
3. Loop `cfg.nb_iterations`:
   - `collect_trajectory()` → `TrajectoryData`
   - Optionally reuse previous trajectory
   - `value_update()` (if separate)
   - `ppo_update()`
   - Save checkpoint every 5 iterations
   - Evaluate every 100 iterations

`scripts/eval.py`: similar but greedy, no updates.
