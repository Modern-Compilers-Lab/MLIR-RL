# Thesis Proposal: Advanced RL-based Automatic Code Optimization in MLIR

## 1. Baseline Architecture Overview

The current system utilizes a structured approach to optimize `Linalg` and `Affine` operations in MLIR:

- **State Representation:** Flat vectors representing loop bounds and access matrices, processed via an LSTM-based encoder.
- **Agent:** PPO (Proximal Policy Optimization) making discrete decisions on a per-operation basis.
- **Action Space:** Tiling, Parallelization, Fusion, Interchange, and Vectorization.
- **Traversal:** Reverse traversal (Consumer $\rightarrow$ Producer) of the computation graph.

---

## 2. Proposed Novelties

### Novelty 1: Hardware-Aware Observation Space

**The Problem:** The current agent is "blind" to the hardware it is optimizing for. A policy trained on a 4-core laptop might be suboptimal for a 64-core server.
**The Solution:** Inject hardware specifications directly into the observation vector.

- **Features to add:** L1/L2/L3 cache sizes (in KB), number of physical vs. logical cores, SIMD vector width (e.g., 256 for AVX2, 512 for AVX-512), and clock speed.
- **Impact:** This enables **Cross-Hardware Portability**. The agent learns to adjust tile sizes and parallelization strategies based on available cache and cores.

### Novelty 2: Deep Loop Nest Parsing (Transformer-based Encoder)

**The Problem:** LSTMs are sequential and struggle with long-range dependencies in complex nested loops.
**The Solution:** Replace the recursive LSTM with a **Transformer Encoder**.

- **Mechanism:** Treat each loop level in a nest as a "token." Use Self-Attention to allow the model to weigh the importance of the outermost loop (for tiling) against the innermost loop (for vectorization) simultaneously.
- **Positional Encoding:** Use structural encoding to maintain the hierarchy of the loop nest (which loop is inside which).

### Novelty 3: Multi-Objective and Shaped Rewards (Already Implemented)

**The Problem:** Speedup is a "sparse" reward (only known at the end of execution) and doesn't account for other factors like energy or memory.
**The Solution:** **Reward Shaping:** Add intermediate rewards based on static analysis:
- **Arithmetic Intensity:** Ops/Byte ratio.
- **Vectorization Ratio:** Percentage of loops successfully vectorized.

- **Multi-Objective:** Create a weighted reward: $R = w_1(Speedup) + w_2(PeakMemoryUsage)$. This is critical for edge computing where memory is constrained.

**Status:** This novelty has been implemented and integrated into the training pipeline (V4.5+).

**Reward Design Lessons (V4.6/7/8):**
- **Shaped reward magnitude:** Must be ≤10% of terminal reward magnitude. Original V4.5 had shaped reward dominating terminal speedup reward by 20×, causing the agent to optimize static heuristics (parallel ratio, vectorizability) rather than actual execution time.
- **Terminal reward anchor:** `log10(speedup)` provides ±0.3-0.5 per benchmark — shaped reward must be scaled to this range (`reward_shaping_scale=0.05`).
- **Slowdown penalty:** Zero intermediate rewards when speedup < 1.0 to prevent the agent from being rewarded for making code slower.
- **Vectorization bonus:** Must scale with `reward_shaping_scale`. In V4.5, it was added after scaling (bug), inflating shaped reward.
- **Hardware core count:** Changed from `os.cpu_count()` (physical) to `SLURM_CPUS_PER_TASK` to reflect allocated resources on HPC clusters.
- **Execution timeout:** Min reduced from 30s to 2s (`MIN_EXEC_TIMEOUT` env var), multiplier 10x → 5x. Bad schedules fail fast; 2s safety kill is sufficient since a good schedule runs well under baseline time.

---

## 3. Future Work (V4+)

### Reward-Fixed Training Runs (V4.6, V4.7, V4.8) — Completed
**Status:** Trained on 18-model new dataset (Bergamo HPC). All three share the V4.5 implementation package (`rl_autoschedular_v4_5`) with corrected reward shaping and execution timeout improvements.

| Version | Transformer | Sched/Iter | Best Avg Speedup |
|---------|------------|-----------|-----------------|
| V4.6 | Classic (d=256, h=8, L=3, ffn=1024) | ~97s | 1.82x |
| V4.7 | Small (d=64, h=2, L=2, ffn=128) | ~62s | 2.59x (early), regressed to 1.42x |
| V4.8 | Classic (d=256, h=8, L=3, ffn=1024) | ~97s | 1.98x |

All three use identical reward fixes (scale=0.05, clip=0.1, vec_bonus=0.0, slowdown penalty).

See [`docs/VERSIONS.md`](VERSIONS.md) for full details on each version's fixes, results, and lessons learned.

### Novelty 4: Full Graph Observation (Future Work - V5)

**The Problem:** The current agent optimizes operations in isolation, traversing the computation graph in reverse order (Consumer → Producer). This limits the agent's ability to make globally optimal scheduling decisions that depend on the entire model structure.
**The Solution:** Enhance the observation space to include the full computation graph structure as input to the agent.

- **Mechanism:** Instead of per-operation features, construct a graph neural network (GNN) representation of the entire model dependency graph.
- **Impact:** The agent gains context about downstream operations, enabling scheduling decisions that account for data reuse patterns and cross-operation optimization opportunities.

### Novelty 5: Expanded Transformation Action Space (Future Work - V6)

**The Problem:** The current set of actions is limited to high-level loop transformations (Tiling, Parallelization, Fusion, Interchange, Vectorization). The agent cannot express schedules that combine high-level tiling with low-level memory layout changes or instruction-level parallelism optimizations.
**The Solution:** Implement finer-grained transformations available in the MLIR Transform Dialect. This is planned for future work (e.g., **`rl_autoschedular_v6`**):

- **Pad (`P`):** Pads operation dimensions to multiples of powers of 2, ensuring aligned memory accesses and efficient vectorization.
- **Pack (`PK`):** Reorganizes data into blocked/tiled layouts using `transform.structured.pack`, improving cache locality for tiled access patterns.
- **Unroll (`U`):** Tiles loops and then unrolls them with `transform.loop.unroll`, exposing instruction-level parallelism and reducing loop overhead.

See [`docs/Novelties/v5_action_space_expansion.md`](Novelties/v5_action_space_expansion.md) for full planned implementation details.

### Novelty 6: Guided Search Strategy (Beam Search / MCTS) (Future Work - V6)

**The Problem:** The RL agent currently suffers from "Greedy Brittleness" — it makes a single "greedy" choice per step. If one choice is wrong, the entire schedule fails because it cannot backtrack.
**The Solution:** Implement a search layer during the inference/evaluation phase. _Note: This is currently not yet implemented and is planned for future work (e.g., `rl_autoschedular_v6`)._

- **Beam Search:** Instead of picking the top action, keep the $K$ most likely transformation sequences and evaluate them all.
- **Monte Carlo Tree Search (MCTS):** Use the RL policy as a "prior" to guide an MCTS exploration. This helps the agent find deep optimization sequences that a standard greedy policy would miss.

### Novelty 7: Meta-Learning for Fast Adaptation (MAML) (Future Work)

**The Problem:** Training an RL agent from scratch for every new dialect or hardware is expensive.
**The Solution:** Use **Model-Agnostic Meta-Learning (MAML)**.

- **Goal:** Train the agent on a wide variety of "tasks" (different kernels like MatMul, Convolutions, Softmax).
- **Outcome:** The agent develops a "meta-policy" that can adapt to a completely new, unseen kernel with only 5–10 optimization steps.

