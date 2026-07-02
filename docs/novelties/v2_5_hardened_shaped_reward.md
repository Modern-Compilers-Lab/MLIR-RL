# V2.5: Hardened Shaped Reward (Fair Baseline)

## Overview

Version 2.5 is a "Hardened" version of the original **V2 (Shaped Reward)** agent. It is designed to serve as a **fair baseline** for head-to-head comparison against **V4.5 (Integrated Robust)**. 

By porting the reliability and safety engineering from V4.5 back to the V2 architecture, we isolate the performance gains of the V4.5 novelties (Transformer + Hardware-Awareness) from the improvements caused simply by a more stable execution platform.

## Rationale for V2.5

The original V2 implementation achieved significant speedups but suffered from a high failure rate (~36%) due to native crashes and the "gambler's incentive" (earning shaped rewards on code that ultimately failed). 

V2.5 eliminates these engineering-level failures while keeping the **Core Algorithm** identical to V2:
1.  **Encoder:** Standard LSTM (no Transformer).
2.  **Observation:** Code features only (no Hardware-Awareness).
3.  **Reward:** Dense Shaped Reward (from V2).

## Ported Hardening (The 4 Pillars)

V2.5 includes all four reliability pillars introduced in V4.5:

### 1. Process Isolation
All MLIR transformations and binary executions are wrapped in `multiprocessing.Process`. Native SIGABRTs from MLIR bindings no longer kill the RL training loop.

### 2. Success-Contingent Reward Negation
Intermediate shaped rewards are now strictly **provisional**. If the final execution fails, all rewards for that episode are zeroed out. This forces the LSTM agent to prioritize runnable optimizations, matching the safety constraints of V4.5.

### 3. Dynamic Timeouts
Replaces the static 30s timeout with a profiling-based margin (10x baseline time, max 300s). This ensures V2.5 isn't unfairly penalized for optimizing large kernels that take longer to profile.

### 4. Stability Rails (Action Masking)
Proactively masks out "risky" action patterns:
-   **Sequence Boundary:** The agent is restricted to terminal actions (`NT` or `V`) as it reaches the end of the predefined `order` sequence or approaches the environment's `truncate` limit.
-   **Depth:** Vectorization is forbidden if the loop-nest depth is > 6.

## Comparison Table

| Feature | V2 (Original) | V2.5 (Hardened) | V4.5 (Integrated) |
| :--- | :--- | :--- | :--- |
| **Encoder** | LSTM | **LSTM** | Transformer |
| **Hardware-Aware** | No | **No** | Yes |
| **Shaped Reward** | Yes | **Yes (Negation-Hardened)** | Yes (Negation-Hardened) |
| **Execution** | In-process | **Isolated Subprocess** | Isolated Subprocess |
| **Stability Rails** | No | **Yes** | Yes |

## Configuration

V2.5 uses the `rl_autoschedular_v2_5` package and targets `results/experiment3`.

```json
{
  "implementation": "rl_autoschedular_v2_5",
  "results_dir": "results/experiment3",
  "reward_shaping_enabled": true
}
```

## Significance

Evaluating V2.5 alongside V4.5 in `experiment3` will reveal the true "Novelty Value" of the Transformer and Hardware features. If V4.5 significantly outperforms V2.5, we can attribute that gain to the architectural integration rather than just better reliability.
