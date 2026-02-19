# Global System Description — MLIR-RL Automatic Action Synthesis Framework

## 1. Purpose of the System

This system is designed to **automatically synthesize and validate reinforcement-learning action spaces for MLIR compiler optimization**, with a primary focus on structured numerical kernels such as **matrix multiplication, convolution, and generic loop-based tensor computations**.

The core challenge addressed is that **manual action-space design for compiler RL is extremely expensive, brittle, and slow**, due to:
- complex legality constraints,
- subtle interactions between transformations,
- and the need for parameterized, composable actions.

This system replaces manual action design with a **multi-agent LLM-driven pipeline** that:
1. reasons about *what optimizations are useful*,
2. synthesizes *executable, parameterized actions* with contracts,
3. verifies *composability and correctness* before integrating actions into an RL environment.

## 2. High-Level Architecture

The system is composed of **three conceptual layers**, each implemented by one or more specialized agents:

### Layer 1 — Optimization Reasoning Agent (Intent & Action Enumeration)

Role: Acts as a **compiler optimization engineer**.

Responsibility:
- Analyze a given MLIR payload (code template).
- Reason *abstractly* about all optimization opportunities that could improve performance.
- Enumerate **atomic optimization transformations** (e.g., tiling, vectorization, fusion).
- Group these transformations under **optimization intents** (e.g., cache locality, SIMD exposure).

Key Properties:
- Does **not** write MLIR Transform dialect code.
- Does **not** think in terms of implementation or debugging.
- Operates at the level of *what transformations exist* and *why they help*.

Output Artifact:
- A structured metadata containing:
  * prioritized optimization intents,
  * atomic transformations per intent,
  * decision rationale for each.

This output is **metadata**, not executable actions.

### Layer 2 — Action Synthesis & Analysis Agents (Executable Action Definition)

Role: Acts as a **compiler transformation engineer**.

Responsibility:
- Take *one atomic transformation* suggested by Layer 1.
- Synthesize a **fully executable RL action**, represented as:
  * a parameterized Python action,
  * embedding MLIR Transform dialect code where needed.
- Define the action as a **contract**, not a script.
- Consult MLIR documentation when needed for correct dialect usage.

Each Action Must Define:
- **Parameters**: tunable knobs exposed to the RL agent.
- **Preconditions**: executable Python checks deciding applicability.
- **Preprocessing**: optional canonicalization or preparation logic.
- **Transform implementation**: parameterized transform dialect logic.
- **Postconditions**: executable Python checks validating the result.
- **Failure semantics**: classification of “no-op”, “not applicable”, “invalid params”, or “transform failure”.

Key Properties:
- Preconditions, postconditions, and logic are **Python code**, not declarative rules.
- Actions must be **stable, reusable, and parameterizable**.
- Actions are independent artifacts that can be injected directly into an RL environment.

Output Artifact
- An **Action Package** (JSON + embedded Python code) that can be loaded and executed without human intervention.

### Layer 3 — Schedule & Interaction Verification Agent

Role: Acts as an **integration and validation agent**.

Responsibility:
- Combine synthesized actions into **sequences (schedules)**.
- Verify that actions:
  * execute without crashing,
  * preserve IR validity,
  * compose correctly with one another.
- Discover **ordering constraints**, conflicts, and enabling relationships.

Key Properties:
- Performance optimality is *not* the primary concern here.
- Focus is on **correctness, composability, and robustness**.
- Failures are minimized to short, reproducible sequences.

Output Artifact:
- Validated action sequences.
- Discovered ordering constraints and incompatibilities.
- Feedback to Layer-2 for refining action definitions.

## 3. Core Design Principles (Shared Across All Agents)

### **Action != Script**
  An action is a **parameterized transformation with a contract**, not an ad-hoc script.

### **Separation of Concerns**
  - Layer 1: *what* should exist.
  - Layer 2: *how* it is implemented.
  - Layer 3: *whether it composes correctly*.

### **Composability First**
  Actions must be safe to chain; failure modes must be explicit.

### **RL-Friendliness**
  Action spaces should be:
  - discrete at the macro level,
  - parameterized at the micro level,
  - maskable based on preconditions.

### **Generalizability**
  While MLIR Transform dialect is the initial target, the architecture is compiler-agnostic:
  the abstraction is "compiler action with executable contract."

## 4. Relationship to Reinforcement Learning

- The final output of this system is a **validated, parameterized action space**.
- RL policies operate over:
  * macro actions (e.g., Tile, Vectorize),
  * conditional parameter subspaces.
- Hierarchical or masked policies are expected.
- Ordering constraints discovered by Layer-3 may be enforced via masks or curriculum.

## 5. Expected Behavior of Agents

All agents:
- Are aware of the **full system architecture**.
- Know **which layer they belong to**.
- Produce outputs strictly scoped to their role.
- Do **not** assume responsibilities of other layers.

Violating layer boundaries (e.g., Layer-1 writing transform code) is considered incorrect behavior.

## 6. Targets and Assumptions

### Compiler Target
- The target compiler infrastructure is **MLIR**.
- Transformations primarily operate on structured MLIR (e.g., `linalg.*`, `scf.*`, `affine.*`) and their progressively lowered forms.
- MLIR Transform dialect is the primary mechanism for expressing and applying transformations in Layer 2.

### Hardware Target (This Machine / Default Target)
- Primary target: **HPC-class CPU** — specifically **Intel Xeon E5-2680 v4 (Broadwell-class)**.
- Topology:
  * **28 physical cores** (2 sockets x 14 cores), **2 NUMA nodes**.
  * **No SMT / Hyper-threading disabled** (threads per core = 1).
- SIMD / ISA capabilities:
  * **AVX2 + FMA available**.
  * **No AVX-512** (do not assume AVX-512 vector widths, masks, or AVX-512-specific lowering).
  * Practical vector lane guidance:
    - FP32: typically 8 lanes per vector (256-bit)
    - FP64: typically 4 lanes per vector (256-bit)
- Cache hierarchy characteristics:
  * L1d ~32KB per core, L2 ~256KB per core, shared L3 per socket (~tens of MB).
- Optimization emphasis for this hardware:
  * **cache-aware tiling** (L1/L2-friendly) and **SIMD vectorization** (AVX2-level),
  * **coarse-grain parallelism** over outer loops (avoid oversubscription),
  * **NUMA awareness** for large tensors and multi-socket scaling,
  * caution with overly aggressive fusion/unrolling due to **register pressure** and potential spills.

### Workload Domain
- Primary workload domain: **machine-learning kernels** on CPUs, especially:
  * matrix multiplication / tensor contractions,
  * convolution (common layouts such as NCHW/FCHW and variants),
  * attention-family kernels (QKV projections, softmax, attention matmul patterns),
  * other loop-nest-dominated linear algebra kernels.
- Workloads are typically compute-intensive and dominated by regular loop nests; performance is sensitive to tiling, fusion, vectorization, and memory layout.

### Non-Goals (by Default)
- GPU-specific optimizations (warps/blocks/shared memory) are **out of scope**.
- Irregular control-heavy code is not a primary focus.
- Algorithmic changes that alter numerical meaning are not considered.

### Guidance for All Agents
- Prefer transformations commonly used in high-performance CPU ML and linear algebra libraries.
- Assume correctness means preserving program semantics and numerical equivalence.
- When uncertain, prefer robust, general-purpose CPU optimizations over fragile, microarchitecture-specific tricks.

## 7. Action Implementation Snapshot (Illustrative Example)

This section provides a **non-normative** example to clarify what the system means by an "action" in practice.
It is included to align all agents on the intended abstraction:
- **Layer 1** proposes *what* transformations exist (ideas only).
- **Layer 2** turns a transformation idea into a **parameterized executable action** (Python + MLIR Transform dialect).
- **Layer 3** validates that actions compose and execute robustly in sequences.

### What an Action Typically Looks Like (Conceptual)
In our current implementation, an action is commonly represented as a **parameterized Python function** that:
1. receives the current MLIR payload IR as text,
2. performs lightweight parameter handling / preprocessing (if needed),
3. constructs a parameterized MLIR Transform dialect snippet (template injection),
4. executes it using an internal runner (e.g., `__run_transform_code`),
5. returns the transformed MLIR payload IR (or a classified no-op / failure).

### Example: Parameterized Tiling Action (Illustrative)
```python
def transform_tile(code: str, operation_tag: str, tiling_sizes: list[int]):
    # If tiling sizes are all zeros, treat as no-op
    if all([a == 0 for a in tiling_sizes]):
        return code

    n_loops = sum([s != 0 for s in tiling_sizes])
    r = ', '.join(['!transform.any_op'] * n_loops)
    assert n_loops > 0, "No loops to tile"

    transform_code = (
        f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
        f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
        f'    %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1'
        f' : (!transform.any_op) -> !transform.any_op\n'
        f'    %tiled_op_{operation_tag}, %loops:{n_loops} = transform.structured.tile_using_for %op_{operation_tag}'
        f' tile_sizes {str(tiling_sizes)} : (!transform.any_op) -> (!transform.any_op, {r})\n'
        f'    transform.yield\n'
        f'  }}\n'
        f'}}\n'
    )

    return __run_transform_code(code, transform_code)

Important Notes (for all agents)
The example above is illustrative, not a strict requirement; it conveys the current direction:
- actions are parameterized,
- implementable as executable units,
- and expressed through MLIR Transform dialect where applicable.
