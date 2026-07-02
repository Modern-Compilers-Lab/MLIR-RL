# Full-Model RL: Design for True End-to-End Support

## 1. Problem Scope

The current agent schedules one operation at a time (bag-of-per-op-schedules). Full-model RL needs:

- A **global state** that captures the entire computation graph (not just one op + its producer)
- An **action space** that schedules multiple ops jointly (fusion, loop fission, inter-op tiling)
- A **training loop** that works on full `.mlir` files (not extracted blocks)
- **Infrastructure** that scales to 950MB files without timing out

This document covers all changes required to support arbitrary `.mlir` files as input for both evaluation and training.

---

## 2. State: Global Graph Representation

### 2.1 What Exists

`state.py::BenchmarkFeatures` is per-operation:
```
operations: dict[str, OperationFeatures]  # one entry per op tag
operation_tags: list[str]                # insertion order
```

Each `OperationFeatures` knows its own loops, memory access, arithmetic — but nothing about sibling ops (same-level consumers/producers at a distance >1) or the global graph shape.

### 2.2 What Is Needed

Replace `BenchmarkFeatures` with a `FullModelFeatures` that captures the **entire computation graph**:

```python
@dataclass
class FullModelFeatures:
    graph: nx.DiGraph  # nodes=ops, edges=producer→consumer
    op_features: dict[str, OperationFeatures]
    global_stats: FullModelStats

@dataclass
class FullModelStats:
    num_operations: int
    num_loops_total: int
    max_depth: int
    total_arithmetic_ops: int
    graph_diameter: int
    num_connected_components: int
    has_cyclic_dependencies: bool  # should not happen in linalg, but check
    memory_footprint_bytes: int
```

Constructed once per model (not once per op) during training/eval. The AST dumper already outputs the `#BEGIN_GRAPH` section (producer→consumer edges) — parse it into a NetworkX `DiGraph` once and reuse.

### 2.3 Observation Tensor

Current: per-op fixed-size tensor (OpFeatures + ProducerOpFeatures + ActionHistory + ActionMask).

Future: **three-part observation**:

1. **Local (per-op) features** — same as today's OpFeatures + ProducerOpFeatures (256 dims)
2. **Global features** — `FullModelStats` encoded as 32-dim vector (normalized)
3. **Graph context** — embedding of the operation's **k-hop neighborhood** in the computation graph:
   - Use GNN (GATv2 or GraphSAGE, 2-3 layers) to produce 128-dim node embeddings
   - For the current target op, concatenate its embedding + neighbor embeddings (mean/sum pool)
   - The GNN weights are trained end-to-end with the policy head

This gives the agent awareness of what other ops exist, their types, and data flow patterns.

### 2.4 Files to Modify

| File | Change |
|------|--------|
| `rl_autoschedular_v*/state.py` | Add `FullModelFeatures`, `FullModelStats` dataclasses |
| `rl_autoschedular_v*/state.py` | Add `extract_full_model_features(code: str) -> FullModelFeatures` |
| `rl_autoschedular_v*/observation.py` | New `Observation.from_full_model_state(...)` with GNN encoder |
| `rl_autoschedular_v*/model.py` | Add GNN encoder branch (GATv2Conv), merge with MLP policy head |

---

## 3. Action Space: Multi-Op & Global Transforms

### 3.1 What Exists

6 per-op actions (NoTransformation, Tiling, Interchange, TiledParallelization, TiledFusion, Vectorization). TiledFusion fuses producer→consumer, but the agent chooses it during inference on the consumer op — it never sees both ops simultaneously.

### 3.2 What Is Needed

**New action categories:**

| Action | Scope | Effect |
|--------|-------|--------|
| `FuseChain(op_start, op_end)` | Multi-op | Fuse a chain of N consecutive producers/consumers into one linalg.generic |
| `TileFuseWindow(tile_sizes, window_start, window_end)` | Multi-op | Tile a subgraph, fuse all into the resulting loops |
| `DistributeLoops(loop_depth, target_ops)` | Multi-op | Distribute a loop across a set of ops (loops fission) |
| `GlobalVectorize` | Full model | Apply `transform.structured.vectorize_children_and_apply_patterns` to the entire model (not per-op) |
| `GlobalBufferizeDims(memref_layout)` | Full model | Choose memory layout strategy |
| `OperatorFusion(op1, op2, mode)` | Pair | Fuse two ops element-wise (`fuse_into_containing_op`) |

**Action selection strategy** — hierarchical:

1. **Global actions** (vectorize entire model, bufferize) — chosen once per model
2. **Group actions** (fuse chain, tile-fuse window) — chosen once per connected component in the graph
3. **Local actions** (tile, interchange) — chosen per-op, same as today

The RL policy outputs at 3 levels: a global-level logit, a group-level logit per component, and a local-level logit per op.

### 3.3 Files to Modify

| File | Change |
|------|--------|
| `rl_autoschedular_v*/actions/` | Add `FuseChain`, `TileFuseWindow`, `DistributeLoops`, `GlobalVectorize` action classes |
| `rl_autoschedular_v*/actions/__init__.py` | Register new actions, extend `ActionSpace` to hierarchical (3 levels) |
| `rl_autoschedular_v*/transforms.py` | Add transform dialect generators for each new action |
| `rl_autoschedular_v*/model.py` | Add hierarchical action heads (3 separate `nn.Linear` headers) |

---

## 4. Training Loop: Full-Model PPO

### 4.1 What Exists

`train.py` loads a dataset of block-based `BenchmarkFeatures`, runs PPO on one op at a time. Each "episode" is a single op's schedule sequence (4-6 steps). Reward = speedup vs baseline.

### 4.2 What Is Needed

**Each training episode = one full model**, consisting of:

1. **State**: `FullModelFeatures` (parsed once per episode, cached)
2. **Rollout**: For each op in the model (in topological order), choose a local action. Additionally choose 0-1 group action per component and optionally a global action.
3. **Reward**: `(baseline_time / optimized_time) - 1` measured on the FULL model (not per-op)
4. **Credit assignment**: Multi-step returns across the entire model. An op's schedule might hurt individually but enable global fusion that nets positive. Standard PPO handles this via advantage estimation over the full episode.

**Episode budget constraints** (to keep training feasible):

- Max 50 transform dialect operations per full-model episode (prevents explosion)
- Action mask reduces local action choices when budget is tight

### 4.3 Reward Shaping

Same V4.5 rules but at full-model granularity:
- Success-contingent: intermediate rewards zeroed if final code crashes
- Process isolation: entire model evaluation in subprocess with `EXEC_TIMEOUT` (longer, e.g., 4h)
- Failure penalty: `-10.0` if the optimized model crashes or timeouts

### 4.4 Training Cost & Mitigation

A single full-model inference for gpt2 (950MB, 765 ops) costs ~30s for parsing + seconds per action. Even with batched transforms, training on 1000 episodes × 200 PPO updates = prohibitive.

**Mitigations:**

| Technique | Savings |
|-----------|---------|
| Train on small models first (< 50 ops, < 50MB) | Reduces per-episode time 20x |
| Use block-based pretraining → fine-tune on full models | Starts from a useful policy, requires 10x fewer episodes |
| Gradient checkpointing + gradient accumulation | Fits full-model observation on 1 GPU |
| Cache `FullModelFeatures` and `Module.parse()` output | Avoids re-parsing same model in same epoch |
| Only do full-model PPO update every 50-100 steps (not per action) | Reduces training calls 10x |

**Expected wall-clock for full-model training:** ~500 GPU-hours (vs ~12h for block-based).

### 4.5 Files to Modify

| File | Change |
|------|--------|
| `scripts/train.py` | New `FullModelPPO` training loop, support for hierarchical actions |
| `rl_autoschedular_v*/execution.py` | `Execution.execute_model(code)` — benchmark full model as episode |
| `rl_autoschedular_v*/execution.py` | Per-op reward breakdown logging (for debugging) |
| `scripts/train_fullmodel.sh` | New Slurm script (multi-node, longer timeout) |
| `scripts/eval_fullmodel.sh` | New Slurm eval script |

---

## 5. Infrastructure

### 5.1 AST Dumper

Current: parses `.mlir` file → extracts per-op features + graph once per call.

Changes needed:
- Output graph as **explicit adjacency list** (already does `#BEGIN_GRAPH` — may need edge attributes: operand index, tensor shape)
- Output **tensor shapes and element types** for each edge (for memory-aware decisions)
- Output **loop-carried dependencies** (for sequential vs parallel analysis)
- Keep existing output; add optional `--full-model` flag that emits all of the above

### 5.2 Execution Engine

Current: `Execution.execute_code()` buffers one benchmark at a time.

For full-model:
- `execute_model(code: str, timeout=14400)` — 4h timeout for 950MB models
- The `_measure_with_cmd_fallback` from `optimize_full_model.py` is already correct design but needs to be exposed as the primary (not fallback) option for full-model eval
- **Memory monitoring**: kill subprocess if RSS exceeds 120GB (protects cluster from OOM)

### 5.3 Dataset

Current: `benchmarks_folder_path` = directory of block `.mlir` files.

For full-model:
```
data/full_models/
  resnet18_linalg.mlir
  t5_linalg.mlir
  gpt2_linalg.mlir
  ...
data/full_models_base_times.json   # {model_name: baseline_ns}
data/full_models_train.json        # train split (model names)
data/full_models_eval.json         # eval split
```

The config gets a new field: `full_model_benchmarks_path` (directory) and `full_model_json_file` (baseline times JSON).

### 5.4 Performance Optimizations

| Optimization | Why |
|-------------|-----|
| **MLIR Module caching**: keep `Module` object alive during a training episode | Avoids re-parsing 950MB for every action |
| **Lazy transform generation**: build the combined transform script incrementally as actions are chosen | No need to rebuild the entire script from scratch each step |
| **Checkpoint-resume**: save per-model progress in case of timeout (same pattern as V4.5 markers) | A 4h timeout loses all progress without it |
| **Conda env + GCC-14**: same setup as block-based | Already works, no change needed |
| **Slurm CPU allocation**: 16 cores for full-model (not 4) | 950MB models need more memory for MLIR pass manager |

### 5.5 Files to Modify

| File | Change |
|------|--------|
| `utils/config.py` | Add `full_model_benchmarks_path`, `full_model_json_file`, `full_model_exec_timeout` fields |
| `utils/implementation.py` | Add `BASELINE_PREFIX_FULL` mapping |
| `rl_autoschedular_v*/benchmarks.py` | New `FullModelBenchmarks` class |
| `scripts/optimize_full_model.py` | Already exists — refactor into library (not just a script) |

---

## 6. Implementation Order (Phased)

### Phase 1: Infrastructure (2-3 weeks)
- `state.py`: `FullModelFeatures`, `FullModelStats`, `extract_full_model_features()`
- `benchmarks.py`: `FullModelBenchmarks` class
- `ast_dumper`: `--full-model` flag for enriched output
- `execution.py`: `execute_model()` with 4h timeout + memory monitoring
- Refactor `optimize_full_model.py` into reusable library
- Verify: eval-only on 5 models matches current bag-of-schedules results

### Phase 2: Hierarchical Action Space (2-3 weeks)
- Implement `FuseChain`, `TileFuseWindow`, `DistributeLoops`, `GlobalVectorize` actions
- Implement hierarchical action heads in the model
- Add transform dialect generators for new actions
- Test action correctness on small synthetic full models (2-5 ops)
- Verify: single-model inference with new actions doesn't crash

### Phase 3: GNN Encoder (1-2 weeks)
- Add GATv2Conv layers to the model
- Train GNN as standalone (predict speedup of random scheduling on full models)
- Integrate with policy head
- Verify: observation tensor can encode a 200-op model without context overflow

### Phase 4: PPO on Full Models (3-4 weeks)
- `train.py`: full-model PPO loop (small models only, 25-50 ops)
- Reward computation: full-model execution time
- Credit assignment: PPO advantage over full-episode rewards
- Hyperparameter tuning: learning rate, entropy bonus, clipping
- Verify: agent learns to outperform bag-of-schedules on ResNet18 baseline

### Phase 5: Scale & Production (2-3 weeks)
- Train on medium models (50-200 ops)
- Profile and optimize: where is time spent? Parsing? Transform application? Execution?
- Implement gradient checkpointing for GPU memory
- Add checkpoint-resume for training (MLIR crash recovery)
- Validate on all 20+ full models
- Compare speedups vs block-based baseline + vs bag-of-schedules

**Total estimate: 10-15 weeks** for a PhD-level implementation.

---

## 7. Key Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Full-model training is too slow (950MB parses) | High | Train on small models first, use Module caching, limit episode steps |
| GNN + MLP model is too heavy for PPO on single GPU | Medium | Gradient checkpointing, smaller hidden dim, off-policy replay |
| Global actions don't improve speedup vs bag-of-schedules | Medium | Ablation studies; if no gain, drop global actions and focus on group actions |
| Transform dialect can't express multi-op fusion on arbitrary graphs | Low | MLIR transform dialect supports `fuse_into_containing_op` and `merge_handles` — enough for most patterns |
| 4h timeout insufficient for largest models (gpt2) | Medium | Only train on models that can complete; skip gpt2 during training |
| Memory OOM on 950MB + MLIR IR + transform + JIT | Medium | Use 128GB nodes (`--mem=128G`), monitor via `/usr/bin/time -v` |

---

## 8. Measuring Success

| Metric | Current | Target |
|--------|---------|--------|
| Speedup on small models (ResNet18, VGG11) | 1.83x, 2.13x (bag) | 2.5x, 3.0x (hierarchical) |
| Speedup on medium models (T5, LSTM) | 10.07x, 4.53x (bag) | 12x, 6x (hierarchical) |
| Training time per episode (ResNet18) | N/A (block-based) | < 30 min |
| GPU hours to converge | 12h (block-based) | 500h (full-model) |
| Fraction of models with scheduled speedup > 1.0x | 9/9 (bag, ckpt 715) | 18/20 (hierarchical) |
| Models whose training doesn't crash MLIR | N/A (block-based) | 80%+ |
