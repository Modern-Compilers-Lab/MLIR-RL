# V4 Failure Investigation & Reliability Report

This document details the investigation into why the integrated **V4** model showed significant reliability regressions compared to the **V3** model, and outlines the strategy to reach a near-zero failure rate in the **V4.5** implementation.

## 1. Global Performance & Reliability Summary

The following table summarizes the evaluation performance and failure rates across all major versions on the full benchmark suite (3,014 benchmarks).

| Version | Key Novelty | Arithmetic Mean Speedup | Geometric Mean Speedup | Total Failures (Speedup ≤ 1.0x) | Failure Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **V0 (Baseline)** | MDP + PPO + LSTM | 1.64x | 1.08x | 1700 | 56.4% |
| **V1** | + Hardware-Aware Obs. | 2.45x | 1.12x | 813 | 27.0% |
| **V2** | + Shaped Rewards | **9.23x** | **1.45x** | 1084 | 36.0% |
| **V3** | + Transformer Encoder | 2.90x | 1.10x | **6** | **0.2%** |
| **V4 (Integrated)** | **All Combined** | 2.52x | 1.14x | 1507 | 50.1% |

### Failed Benchmarks by Type (V0 - V4)

| Model Family | V0 | V1 | V2 | V3 | V4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Albert** | 116 | 115 | 111 | 0 | 105 |
| **Bart** | 64 | 65 | 64 | 0 | 58 |
| **Bert** | 107 | 108 | 103 | 0 | 104 |
| **Convnext Tiny** | 81 | 67 | 66 | 0 | 67 |
| **Deberta** | 96 | 84 | 95 | 0 | 88 |
| **Densenet121** | 25 | 3 | 24 | 0 | 24 |
| **Distilbert** | 53 | 5 | 53 | 0 | 58 |
| **Efficientnet B0** | 102 | 30 | 59 | 1 | 73 |
| **GPT2** | 88 | 11 | 83 | 2 | 81 |
| **Mobilenet V3** | 82 | 37 | 57 | 1 | 62 |
| **Resnet50** | 51 | 18 | 30 | 0 | 47 |
| **Resnext50** | 48 | 16 | 31 | 0 | 53 |
| **T5** | 32 | 9 | 29 | 0 | 24 |
| **Vit B 16** | 104 | 15 | 57 | 2 | 102 |
| **Synthetic/Other** | 581 | 221 | 175 | 0 | 494 |

---

## 2. Comparative Reliability Summary (V3 vs V4)

| Metric | V3 (Transformer) | V4 (Integrated) |
| :--- | :---: | :---: |
| **Failure Rate (Speedup ≤ 1.0x)** | **0.2%** | **50.1%** |
| **Stability Strategy** | Safety-First | High-Risk, High-Reward |

---

## 3. Why V3 was so much more reliable?

V3 utilizes the **Transformer Encoder** without the intermediate **Shaped Rewards** found in V2 and V4.

*   **Pure Sparse Incentive:** V3 only receives a non-zero reward at the end of a successful execution. If a transformation sequence crashes or times out, the agent receives a massive penalty (-5.0 or -20.0) and zero positive feedback.
*   **Safety-First Emergence:** Without intermediate "partial credit," the agent learns that the only way to earn points is to ensure the code compiles and runs correctly. The Transformer provides a high-fidelity representation of the loop nest, allowing the agent to consistently identify "safe" transformation boundaries.
*   **Consistency over Aggression:** V3 avoids extremely aggressive tiling or vectorization that might trigger MLIR edge-case bugs because the risk of a -20.0 penalty far outweighs the potential marginal speedup gains.

## 4. Why V4 is failing so much?

V4 integrates **Shaped Rewards**, which provide intermediate payoffs for improving arithmetic intensity or parallel loops, even if the final result is unstable.

*   **Risk-Incentive Misalignment:** The agent receives "partial credit" during the transformation steps. If the shaped reward is sufficiently positive (e.g., +2.0 for a large tiling factor), it can partially offset the failure penalty. This encourages the agent to gamble on unstable schedules.
*   **The "Explosion" of Complexity:** By combining the powerful Transformer encoder with risky incentives, the agent discovers extreme optimization patterns. While these occasionally result in massive speedups (up to 350x), they frequently trigger native MLIR crashes or execution timeouts.
*   **Masking by Isolation:** The process isolation implemented to prevent crashes converts what would be a "hard crash" into a "silent failure" (1.0x speedup), which the agent may not be learning to avoid aggressively enough.

---

## 5. Path to 0% Failure Rate (Strategy for V4.5)

To achieve near-zero failures while maintaining V4's peak performance, the following technical fixes are implemented in **V4.5**:

### A. Success-Contingent Reward Negation (Provisional Rewards)
*   **The Logic:** Intermediate shaped rewards must be treated as **provisional**.
*   **Implementation:** 
    1.  Modify `Env.__apply_sequence` to track a cumulative `provisional_shaped_reward`.
    2.  If the final execution succeeds (`exec_succeeded == True`), the rewards are granted as currently implemented.
    3.  If the execution fails or times out, the script must **negate all intermediate rewards** (setting them to 0.0) and only apply the final `-20.0` penalty.
*   **Impact:** Removes the "gambler's incentive." The agent can no longer "break-even" by applying many risky but "high-intensity" transforms that ultimately fail.

### B. Execution-Time Profiling Safeguard (Dynamic Timeouts)
*   **The Logic:** Prevent extremely slow code from being treated as a successful 1.0x speedup.
*   **Implementation:**
    1.  The `Execution` engine now receives the benchmark's **original unoptimized time** (`root_exec_time`).
    2.  It calculates a dynamic timeout: `timeout_s = min(300, max(30, int((root_exec_time / 1e9) * 10)))`.
    3.  This allows up to 10x slowdown margin for profiling, with a 5-minute hard cap.
*   **Impact:** Prevents "false positive" failures where a valid optimization was simply slow to compile/run, while still killing infinite loops.

### C. Physical Boundary Safeguards (Depth Limit)
*   **The Problem:** The MLIR Python bindings have a known diagnostic buffer overflow bug that triggers a native SIGABRT when vectorizing loops nested deeper than 6 levels.
*   **V4.5 Implementation:** `Vectorization` is explicitly forbidden in `ActionSpace.action_mask` if the current loop nest depth is greater than 6.
*   **Impact:** Proactively avoids the most common cause of native interpreter crashes, allowing training to proceed without interruption on deep loop nests.

### D. Multi-Engine Fallback (CMD-Line Execution)
*   **The Problem:** Sometimes the Python bindings crash for reasons unrelated to the optimization.
*   **V4.5 Implementation:** If the isolated Python execution fails, the engine automatically retries using the standalone `mlir-cpu-runner` binary.
*   **Impact:** Ensures that execution failures are only reported if the MLIR code is truly unrunnable.
