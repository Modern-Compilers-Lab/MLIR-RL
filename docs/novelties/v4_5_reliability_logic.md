# V4.5 Reliability Engineering: Deep-Dive Insights

This document summarizes the technical rationale and architectural insights behind the "Hardening" features in V4.5, specifically addressing the trade-off between pure RL exploration and compiler stability.

---

## 1. Dynamic Timeouts (The "10x Tailored Deadline")

**Insight:** Static timeouts (e.g., 30s) are inherently unfair to large, complex benchmarks. 

*   **Source:** The baseline execution time (`root_exec_time`) is taken from the **V0 (Unoptimized)** results.
*   **The 10x Margin:** We allow the optimized code to run up to 10x slower than the unoptimized baseline (capped at 300s).
*   **Why allow "Slow" code?**
    1.  **Stronger Gradients:** In RL, knowing that an action made the code "3x slower" provides a much richer mathematical signal to PPO than simply "Timed Out." It turns a binary failure into a measurable penalty.
    2.  **JIT Overhead:** Optimizing code (tiling, fusion) increases the complexity of the MLIR source. This requires the LLVM backend to work harder during Just-In-Time (JIT) compilation. A static timeout often kills a valid optimization simply because the compiler was taking its time.

## 2. Dynamic Sequence Boundaries (Respecting the "Order")

**Insight:** Runaway complexity is a primary trigger for native compiler crashes. Instead of hardcoded Python limits, V4.5 treats the **Config as the Source of Truth**.

*   **Rationale:** The original `ActionSpace` already enforces the length of the `order` list. If an agent reaches the end of the predefined strategy, the system throws an exception unless it stops.
*   **Safety Buffer:** The environment's `truncate` setting acts as an absolute hard wall. If the agent somehow exceeds the `order` length, `truncate` forcefully ends the episode.
*   **Implementation:** V4.5 keeps the `ActionSpace` logic clean and baseline-compatible, relying on these existing configuration-driven boundaries rather than adding redundant hardcoded masking for sequence length.

## 3. Physical Boundary Safeguards (Depth Limit)

**Insight:** You cannot "train" an agent to ignore a bug in the underlying C++ compiler if that bug causes a catastrophic crash.

*   **Environmental Context:** This bug is a **native C++ limitation** in the MLIR Python bindings (diagnostic buffer overflow). It exists in the shared environment and therefore **affects all versions (V0 - V4.5).**
*   **Why V2 and V4 are most affected:** The introduction of **Shaped Rewards** encouraged "Greedy Exploration." The agent discovered that creating extremely deep loop nests maximized intermediate rewards, inadvertently driving itself into the "Crash Zone" (depth > 6).
*   **Safe Exploration:** In V4.5 and V2.5, we explicitly mask `Vectorization` if the loop nest is too deep. We aren't "cheating" for the agent; we are defining the **Physical Boundaries** of the environment. This keeps the agent focused on finding the best *runnable* schedule within the limits of the current compiler technology.

---

## Summary of Safeguards

| Feature | Logic | Goal |
| :--- | :--- | :--- |
| **Dynamic Timeout** | 10x `root_exec_time` | Fairer feedback & JIT tolerance |
| **Sequence Rail** | Respects `order` & `truncate` | Stay within user-defined strategies |
| **Depth Limit** | Max 6 loops for Vectorization | Avoid native MLIR C++ assertions |
| **Negation** | Zero rewards on failure | Prioritize runnability over partial gains |
