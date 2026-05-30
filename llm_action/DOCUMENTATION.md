# MLIR-RL — `llm_action/` Contribution Documentation

> **Purpose of this document.** This is a single, self-contained reference for the `llm_action/`
> contribution, written to support the **Design and Implementation** chapter of the master's
> thesis. It has two parts:
>
> - **Part A — Technical Reference Dossier**: an exhaustive, citation-ready description of every
>   component, its data flow, and the files that implement it. Use it to look up *facts* while
>   writing (paths, interfaces, defaults, exact behaviours).
> - **Part B — Proposed "Design and Implementation" Chapter Outline**: a section tree with
>   per-section narrative guidance (purpose, what to cover, which Part A material and figures to
>   draw on, and the claim to land). Use it as the *scaffold* for the prose.
>
> Diagrams throughout are ASCII sketches meant to *inspire* the manuscript's real figures, not to
> be pasted verbatim. Each is tagged `[Figure X]` and re-listed in the closing figure map.

---

## One-paragraph abstract of the contribution

Reinforcement learning (RL) has been applied to compiler phase ordering and loop optimization, but
its central bottleneck is rarely the policy: it is the **action space**. Designing a usable RL
action space for a compiler like MLIR is expensive, brittle, and slow — every action must encode
legality constraints, compose safely with the others, and expose RL-friendly parameters, all by
hand. This contribution replaces that manual effort with an **LLM-agent-driven synthesis pipeline**.
A coding agent (Claude, driven through the Claude Code CLI and grounded in a custom **MLIR-Torch
Model Context Protocol (MCP) server**) (1) *enumerates* candidate optimization transformations,
(2) *implements* each one as an executable, parameterized, contract-bearing RL action expressed in
the MLIR Transform dialect, and (3) *explores schedules* to empirically discover how the actions
compose and which schedule shapes are worth learning. The resulting, automatically generated action
space plugs directly into a Gymnasium RL environment whose state encodes the loop nest, whose
**MultiDiscrete** action couples a transformation choice with its parameters, and whose reward is the
measured speedup over a baseline. A **MaskablePPO** agent then learns to optimize MLIR kernels.
The thesis claim is that LLM agents can synthesize *and validate* a compiler RL action space
end-to-end, turning a months-long manual engineering task into an automated, reproducible pipeline.

---

# PART A — Technical Reference Dossier

## A1. System overview & problem framing

### A1.1 The problem: manual RL action-space design is the bottleneck

Casting compiler optimization as an RL problem requires a set of *actions* the policy can apply to
the program. For MLIR loop-nest optimization those actions are transformations such as tiling,
loop interchange, vectorization, parallelization, and promotion. Building this action space by hand
is hard because each action must:

- **Encode legality.** A transformation is only valid on certain IR shapes (e.g. promotion needs
  bufferized memrefs; vectorization needs vector sizes that divide the loop bounds).
- **Compose with the others.** Actions are chained into *schedules*; one action can enable or
  destroy the preconditions of another (e.g. vectorization lowers the `linalg` op away, so nothing
  structure-preserving can follow it).
- **Expose RL-friendly parameters.** Tile sizes, vector widths, thread counts, and permutations must
  be discretized into something a policy can sample without drowning in a combinatorial space.

Doing this manually is **expensive** (senior compiler-engineer time), **error-prone** (subtle
Transform-dialect and handle-invalidation bugs), and **slow to experiment with** (every new action
or kernel family is a fresh round of hand-coding). This is the bottleneck the contribution attacks.

### A1.2 The solution: three integrated worlds

The system integrates three traditionally separate domains into one closed loop:

```
[Figure 1] End-to-end system architecture

        ┌──────────────────────────── LLM AGENT ────────────────────────────┐
        │  Claude (Claude Code CLI, --dangerously-skip-permissions)          │
        │  driven by versioned methodology prompts (resources/prompts/v1/)   │
        │                                                                    │
        │   Layer 1            Layer 2               Layer 3                 │
        │   Enumeration  ───►  Implementation  ───►  Schedule Exploration    │
        │   (what)             (how)                 (does it compose?)      │
        └───────┬───────────────────┬──────────────────────┬────────────────┘
                │ action_enum.json   │ ActionBase classes   │ SCHEDULE_GRAPH
                │ (intents+actions)  │ + registry + mcp.py  │ ACTION_DEPENDENCIES
                ▼                    ▼                      ▼
        ┌──────────────────── COMPILER (MLIR) ───────────────────────────────┐
        │  MLIR Transform dialect transforms + bufferize/lower pass pipeline  │
        │  executed for real timing via the MLIR-Torch MCP / executors        │
        │  PyTorch baselines measured on identical shapes for fair speedup     │
        └─────────────────────────────┬───────────────────────────────────────┘
                                       │ generated, validated action space
                                       ▼
        ┌──────────────────────────── RL (PPO) ──────────────────────────────┐
        │  Gymnasium env (MLIROptEnv): state = loop-nest features + history    │
        │  action = MultiDiscrete (which transform + its parameters)           │
        │  reward = measured speedup over baseline                             │
        │  MaskablePPO actor-critic learns to optimize kernels                 │
        └──────────────────────────────────────────────────────────────────────┘
```

- **Compiler world** — MLIR, with the **Transform dialect** as the mechanism for expressing and
  applying transformations, lowered through a fixed bufferization + LLVM pass pipeline and executed
  for real wall-clock timing.
- **LLM-agent world** — Claude, running agentically via the Claude Code CLI, grounded in the
  compiler and in PyTorch through a custom MCP server, producing the action space.
- **RL world** — a Gymnasium environment and a PPO agent that consume the generated action space.

### A1.3 Core design principles (shared by all agents)

These principles are stated in the shared system description
(`resources/prompts/v1/system_description.md`, and re-embedded in each layer's prompt) and recur
throughout the implementation:

| Principle | Meaning |
|---|---|
| **Action ≠ Script** | An action is a *parameterized transformation with a contract*, not an ad-hoc script. |
| **Separation of concerns** | Layer 1 decides *what* should exist, Layer 2 *how* it is implemented, Layer 3 *whether it composes*. |
| **Composability first** | Actions must be safe to chain; failure modes are explicit booleans. |
| **RL-friendliness** | Discrete at the macro level, parameterized at the micro level, maskable by precondition. |
| **Generalizability** | MLIR is the first target, but the abstraction is "compiler action with executable contract." |

### A1.4 Target assumptions

The shared system description pins the optimization target so the agents reason concretely:

- **Hardware:** Intel Xeon E5-2680 v4 (Broadwell-class), 28 physical cores (2×14, 2 NUMA nodes), no
  SMT; **AVX2 + FMA, no AVX-512** (FP32 ≈ 8 lanes, FP64 ≈ 4 lanes / 256-bit). L1d ≈ 32 KB/core,
  L2 ≈ 256 KB/core, shared L3 per socket. (`N_CORES = 28` in `src/config.py`.)
- **Workload:** ML CPU kernels dominated by regular loop nests — matmul/contractions, convolution,
  pooling, elementwise (add/relu), and composite ML graphs.
- **Emphasis:** cache-aware tiling, AVX2 SIMD vectorization, coarse-grain outer-loop parallelism,
  NUMA awareness; caution with aggressive fusion/unrolling (register pressure).
- **Non-goals:** GPU-specific optimization, irregular control-heavy code, numerically-altering
  algorithmic changes.

---

## A2. The automatic action-generation pipeline (3 layers)

The pipeline is the heart of the contribution. Each layer is a **specialized Claude agent** invoked
once by a SLURM batch script; the script runs the Claude Code CLI with a prompt produced by a Python
generator, which in turn embeds (a) a versioned *methodology spec* and (b) a representation of the
target benchmark set.

```
[Figure 2] Three-layer action-generation pipeline (with artifacts)

 INPUT: one op family's MLIR kernels (data/benchmarks/<set>/train/*.mlir),
        tagged with  tag = "operation_0"  on the op to optimize.

 ┌─ LAYER 1 ─ ENUMERATION ──────────────────────────────────────────────┐
 │ driver:  scripts/claude_enumeration.sh                                │
 │ prompt:  src/prompts/claude_enumeration.py                            │
 │ spec:    resources/prompts/v1/action_enumeration.md                   │
 │ agent role: "Expert MLIR Optimization Engineer" — reasons abstractly  │
 │ OUT  ►  src/actions/v<x>/enumeration/action_enumeration.json          │
 │         + reasoning.md      (intents → macro transformations)         │
 └───────────────────────────────┬───────────────────────────────────────┘
                                  │  catalog of macro actions + action_template
                                  ▼
 ┌─ LAYER 2 ─ IMPLEMENTATION ───────────────────────────────────────────┐
 │ driver:  scripts/claude_implementation.sh                             │
 │ prompt:  src/prompts/claude_implementation.py                         │
 │ spec:    resources/prompts/v1/action_implementation.md                │
 │ agent role: "Expert MLIR Transformation Engineer"                     │
 │ tools:   mlir-tools MCP (transform / execute / measure / docs)        │
 │ OUT  ►  src/actions/v<x>/implementation/<action>.py  (ActionBase)     │
 │         src/actions/v<x>/tests/test_<action>.py                       │
 │         src/actions/v<x>/registry.py  (ACTION_CLASSES)                │
 │         src/actions/v<x>/mcp.py       (per-action MCP server)         │
 │         .mcp.json  entry  rl-action-v<x>                              │
 └───────────────────────────────┬───────────────────────────────────────┘
                                  │  executable, parameterized action set
                                  ▼
 ┌─ LAYER 3 ─ SCHEDULE EXPLORATION ─────────────────────────────────────┐
 │ driver:  scripts/claude_exploration.sh                                │
 │ prompt:  src/prompts/claude_exploration.py                            │
 │ spec:    resources/prompts/v1/schedule_exploration.md                 │
 │ agent role: "Expert MLIR Schedule Exploration Engineer"               │
 │ tools:   rl-action-v<x> MCP  +  mlir-tools MCP   (+ parallel agents)  │
 │ logs  ►  logs/mcp/v<x>/<kernel>_<YYMMDDHHMM>.md                       │
 │ OUT  ►  appends to src/actions/v<x>/registry.py:                      │
 │         SCHEDULE_GRAPH       (allowlist of winning schedule shapes)   │
 │         ACTION_DEPENDENCIES  (denylist of illegal transitions)        │
 └───────────────────────────────────────────────────────────────────────┘

 RESULT: a complete, self-contained action version  src/actions/v<x>/
         ready to be loaded by the RL environment.
```

**Common invocation pattern (all three scripts).** Each SLURM script activates the `mlir` conda
env, optionally connects the MCP servers (`claude /mcp`), and runs:

```bash
claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_<layer>.py $ARGS)"
```

i.e. the *prompt is generated by Python* and the *agent is the Claude Code CLI*. The prompt
generators embed the target benchmark representation via `format_for_prompt(...)` from
`src/data/benchmarks.py` (per family: the template, one full concrete instance, and the
names/shapes of the other instances — a token-budget knob `--limit` caps the listing).

**Versioning.** Every run writes a *new, independent* `src/actions/v<x>/` directory (currently
`v0 … v52`). The prompts enforce a hard rule: an agent may consult only `v0/` (the empty structural
template) and must **not** read other versions, so each synthesis is unbiased. `v0/` is reserved as
the reference skeleton (its files are intentionally empty).

### A2.1 Layer 1 — Action Enumeration (the *what*)

- **Driver / prompt / spec:** `scripts/claude_enumeration.sh` → `src/prompts/claude_enumeration.py`
  → `resources/prompts/v1/action_enumeration.md`.
- **Agent role:** "Expert MLIR Optimization Engineer." It reasons *abstractly* about optimization
  opportunities from a generic **loop-nest** viewpoint. It does **not** write Transform-dialect code,
  parameters, legality, or ordering — those belong to Layers 2 and 3.
- **Process:** read the kernel templates → identify high-level **optimization intents** (e.g. cache
  locality, SIMD exposure, parallel work distribution) → under each intent, enumerate **macro RL
  actions** (e.g. Tiling, LoopInterchange, Vectorization, Parallelization, Promotion, Packing) →
  assign each intent a priority (HIGH/MEDIUM/LOW) and give rationales.
- **Granularity rule:** an action is a *reusable, kernel-agnostic* transformation class. Differences
  that could be expressed as parameters later (which loop, which dimension) must **not** be split
  into separate actions. (Correct: "Tiling". Incorrect: "Tile batch dimension".)
- **`action_template`:** each transformation carries a template string such as
  `Tiling(tile_sizes) OR Tiling(loop_band, tile_sizes)` using only generic loop-nest vocabulary —
  this is the bridge that tells Layer 2 how to parameterize.
- **Input:** the benchmark representation for one op family (template + concrete instance + sibling
  shapes).
- **Output (artifact):** `src/actions/v<x>/enumeration/action_enumeration.json` (validated against
  Pydantic models `Transformation`, `OptimizationIntent`, `ActionEnumeration`) plus `reasoning.md`.
  Constraint: 2–3 intents, each with 2–3 transformations. **This output is metadata, not code.**
- **Note on splitting for parameterization:** the spec explicitly asks the agent to enumerate
  *sequential* and *parallel* vectorization (and tiling-based vs. num-threads parallelization) as
  **separate** macro actions, because they require structurally different preprocessing — a decision
  that propagates into Layers 2 and 3.

### A2.2 Layer 2 — Action Implementation (the *how*)

- **Driver / prompt / spec:** `scripts/claude_implementation.sh` →
  `src/prompts/claude_implementation.py` → `resources/prompts/v1/action_implementation.md`.
- **Agent role:** "Expert MLIR Transformation Engineer." It turns **one** abstract transformation
  into **one** executable, reusable, deterministic RL action.
- **Tools (MCP, self-validation):** the agent grounds and validates each action using the
  `mlir-tools` MCP server through a strict **5-step tool plan** per kernel:
  1. `delegate_documentation_lookup(task)` — ground Transform-dialect op names/handles/attributes;
  2. `execute_mlir_code(original)` — baseline must succeed;
  3. `transform_mlir_code(original, transform_ir)` — output must differ from input;
  4. `execute_mlir_code(transformed)` — must succeed;
  5. `measure_speedup(base, transformed)` — sanity check (not optimization).
- **Memory:** a persistent scratchpad `docs/memory/MEMORY.md` records non-obvious MLIR/Transform
  quirks and their fixes, so later runs suffer fewer re-discoveries.
- **Output (artifacts):** for the new version `v<x>`:
  - `implementation/<action>.py` — a Python class extending `ActionBase` (see A3);
  - `tests/test_<action>.py` — a standalone test that must run end-to-end on the family's kernels
    (using the `test_action` harness, A3.5);
  - `registry.py` — the `ACTION_CLASSES` list;
  - `mcp.py` — a per-action FastMCP server exposing each action as a tool (used by Layer 3);
  - a `.mcp.json` entry named `rl-action-v<x>` (and a switch of the base server to the *minimal*
    variant — see A6).
- **Key contracts the agent must honour** (detailed in A3): the `tag = "operation_0"` targeting and
  re-annotation protocol, execution multiplicity (`unique_execution`), the vectorization safety
  contract (vector size ≤ 2048, rank ≤ 3, sizes divide loop bounds), and the promotion contract
  (bufferize first, re-match handles after bufferization, canonicalize).

### A2.3 Layer 3 — Schedule Exploration (the *does it compose?*)

- **Driver / prompt / spec:** `scripts/claude_exploration.sh` →
  `src/prompts/claude_exploration.py` → `resources/prompts/v1/schedule_exploration.md`.
- **Agent role:** "Expert MLIR Schedule Exploration Engineer." It treats the action set as a black
  box (it may **not** modify action code or write raw Transform IR) and explores how actions compose,
  using only the `rl-action-v<x>` and `mlir-tools` MCP tools.
- **Phased process:**
  - **Phase 0 — baselines:** measure each kernel's unoptimized MLIR time.
  - **Phase 1 — single actions:** apply each action once with a valid probe parameter set; record
    precondition/postcondition/time/speedup and tag preservation.
  - **Phase 2 — pairwise composability (exhaustive):** for every ordered pair (A then B), test
    whether B applies after A; build a full **composability matrix** (every cell must come from an
    actual tool call — "N/A by reasoning" is forbidden).
  - **Phase 3 — multi-step schedule shapes:** chain 3+ actions into distinct *skeletons* (e.g. the
    canonical `parallelize → tile → tile → vectorize`), each measured once.
  - **Phase 4 — selection across cases:** identify the best schedule **shape** per kernel subset.
  - For Phases 3–4 the main agent **spawns parallel sub-agents**, one per kernel-family subset, to
    widen coverage within the time budget, then synthesizes their findings.
- **Crucial scoping rule:** Layer 3 discovers schedule **shapes**, *not* parameters. It uses a single
  divisor-valid probe parameter set per schedule and never sweeps — **parameter tuning is the RL
  policy's job.** A structurally promising shape is kept even if one probe value underperforms.
- **Logs:** human-readable markdown written incrementally to `logs/mcp/v<x>/<kernel>_<datetime>.md`.
- **Output (artifacts, appended to `registry.py`):**
  - **`SCHEDULE_GRAPH`** *(primary deliverable)* — a per-family **allowlist** of high-value schedule
    paths (action-name skeletons, no parameters). It positively guides the policy: at each step the
    policy may only pick an action that *extends* the successfully-applied prefix along one of these
    paths.
  - **`ACTION_DEPENDENCIES`** *(legacy)* — a **denylist**: "if action X has executed, action Y is
    unavailable." Only structural (not parameter-specific) block edges, each grounded in an observed
    Phase-2 failure.
  - The two must be consistent (no `SCHEDULE_GRAPH` path may include a blocked transition).

---

## A3. The action template & contract

Every generated action is a Python class implementing a fixed **contract**. The contract is what
makes an "action ≠ script": it is parameterized, legality-checked, composable, and RL-mappable.

### A3.1 The `ActionBase` interface

Defined in `src/actions/base.py`:

```python
class ActionBase(ABC):
    unique_execution: bool = True            # may this action repeat within one episode?

    # --- transformation contract ---
    @classmethod @abstractmethod
    def parameters(cls) -> dict: ...                          # declared tunable knobs
    @classmethod @abstractmethod
    def precondition(cls, code, params) -> bool: ...          # applicable? (no IR mutation)
    @classmethod @abstractmethod
    def preprocess(cls, code, params) -> str: ...             # optional canonicalization/prep
    @classmethod @abstractmethod
    def implement(cls, code, params) -> str: ...              # build+run Transform IR, return new MLIR
    @classmethod @abstractmethod
    def postcondition(cls, before, after, params) -> bool: ...# succeeded? (rejects no-ops)

    # --- RL parameter interface ---
    @classmethod def params_size(cls) -> int: return 0
    @classmethod def classes_per_slot(cls, n_loops) -> list[int]: return []
    @classmethod def decode_params(cls, raw_slots, n_loops, loop_bounds=None) -> dict: return {}
    @classmethod def valid_param_mask(cls, n_loops, loop_bounds) -> "np.ndarray | None": return None
```

The v0 reference template (`src/actions/v0/implementation/name.py`, `tests/name.py`, `registry.py`,
`mcp.py`) is the empty structural skeleton every Layer-2 run mirrors.

```
[Figure 3] Action contract lifecycle (one RL step)

  current MLIR ──► precondition ──false──► (precondition penalty, action masked)
       │              │ true
       │              ▼
       │           preprocess  (optional canonicalization / prep tiling)
       │              │
       │              ▼
       │           implement ──► build Transform-dialect IR with params
       │                          match tag="operation_0" ─► apply ─► RE-ANNOTATE tag
       │                          run_transform_code(...)  (MLIR Python bindings)
       │              │
       ▼              ▼
   before ───► postcondition(before, after) ──false──► failed-transform penalty
                      │ true
                      ▼
              execute transformed MLIR ─► time ─► reward = f(speedup)
```

### A3.2 The tagging / re-annotation contract

The single op to optimize in every dataset kernel is pre-tagged `tag = "operation_0"`. Every action
must (a) target the op *only* via this tag (never by heuristics or text rewriting), (b) treat a
missing tag as *not applicable*, and (c) **re-annotate** the result with the same tag so the next
action in a schedule can find it. Two tagging categories:

- **Category A — structure-preserving** (output is still a `linalg.*` op: tiling, interchange,
  packing, promotion): tag the resulting `linalg` op directly.
- **Category B — lowering** (the `linalg` op is consumed, replaced by loops + vector ops:
  vectorization): tag the **outermost generated loop** instead. Because
  `transform.structured.match attributes{tag=...}` matches *any* op type, subsequent actions still
  find the tagged `scf.for`. When the tag is consumed and not re-annotated, the episode terminates
  (see A4.5).

### A3.3 Execution multiplicity

The class attribute `unique_execution: bool` declares whether an action may repeat within one
episode, based on its *structural effect*:

- `True` (single-shot): lowering / structure-replacing transforms (Vectorization, Parallelization,
  bufferization, `convert_*`). A second application has no valid target.
- `False` (repeatable): structure-preserving tuning knobs applied at different scopes (multi-level
  Tiling, LoopInterchange at different levels, Promotion of different operands). Capped per episode
  by `MAX_ACTION_EXECUTIONS = 2` (`src/config.py`).

### A3.4 The RL parameter interface (action → MultiDiscrete)

This is how an action exposes itself to the policy. Each action defines its own **vocabulary** and
**slots** (bounded by `MAX_PARAM_SLOTS = 7` and `MAX_VOCAB_SIZE_PER_SLOT = 6`):

- `params_size()` — number of parameter slots.
- `classes_per_slot(n_loops)` — the categorical size of each slot (one independent categorical per
  slot).
- `decode_params(raw_slots, n_loops)` — converts the policy's chosen integers into the `params` dict.
- `valid_param_mask(n_loops, loop_bounds)` — optional per-slot legality mask the env ANDs into the
  policy's distribution *before sampling*, e.g. enforcing that tile/vector sizes **divide** the loop
  bounds (a non-divisible size produces a dynamic remainder loop that downstream vectorization cannot
  lower).

**Worked example — the `Tiling` action** (`src/actions/v10/implementation/tiling.py`):

```python
class Tiling(ActionBase):
    VOCAB = [0, 4, 8, 16, 32]            # 0 = "do not tile this loop"
    unique_execution = False             # multi-level tiling is a legitimate knob

    def precondition: tag present, tile_sizes is a non-all-zero int list
    def implement:    emit transform.structured.tile_using_for with tile_sizes,
                      then re-annotate %tiled_op with tag="operation_0"
    def postcondition: output differs from input AND still contains func.func

    params_size      -> 7                                # MAX_PARAM_SLOTS
    classes_per_slot -> [5,5,...] (len(VOCAB) per loop, up to n_loops)
    decode_params    -> {"tile_sizes": [VOCAB[s] for each slot]}
```

For a 3-loop matmul the policy emits 3 integers in `{0..4}`, decoded to tile sizes drawn from
`[0,4,8,16,32]`. Other actions follow analogous patterns: **permutations** use a *single* slot with
enumerated non-identity candidates (capped at 6) to avoid factorial blow-up; **thread counts** use a
single slot over a fixed set of divisor-safe values; **zero-parameter** actions (fixed lowerings)
return `params_size()=0` (no artificial enable/disable toggles — the policy's *choice* to select the
action is the enable decision).

### A3.5 Standalone testing harness

Each action ships a test that runs it end-to-end on one random instance per family via
`test_action(action_cls, params_per_family)` (`src/actions/test.py`): it loads the benchmark set,
picks an instance, executes the original, applies precondition→implement→postcondition, executes the
transformed code, and prints the measured speedup. The Layer-2 prompt requires every test to
*actually execute* (no precondition-only tests).

### A3.6 Concrete worked version: `v48` (matmul)

`src/actions/v48/registry.py` is a good end-to-end example of a *completed* version. Its action set,
denylist, and allowlist are reproduced in A4.4.

---

## A4. The RL environment

Located in `src/env/`. It is the seam where the generated action space becomes a learnable MDP.

| File | Role |
|---|---|
| `mlir_opt_env.py` | `MLIROptEnv(gym.Env)` — `reset`, `step`, `action_masks`; the MDP itself. |
| `action_registry.py` | Loads `ACTION_CLASSES` / `ACTION_DEPENDENCIES` / `SCHEDULE_GRAPH` for a version. |
| `action_space.py` | Builds the MultiDiscrete space and the masking logic. |
| `state_extractor.py` | Encodes MLIR → observation vector. |
| `env_config.py` | `EnvConfig` dataclass (reward/masking/param modes, etc.). |
| `benchmarks.py` | Loads kernels and pre-measures baselines. |

### A4.1 Gymnasium interface

`MLIROptEnv` implements the standard contract: `reset(seed, options={"benchmark_idx": ...})` →
`(obs, info)`; `step(action)` → `(obs, reward, done, truncated, info)`; and `action_masks()` →
boolean mask consumed by MaskablePPO. An episode optimizes one kernel for up to `MAX_STEPS = 7`
transformation steps.

### A4.2 State / observation

```
[Figure 4] RL MDP and the observation vector

  observation = [ OP-FEATURES (137) | ACTION-HISTORY (max_steps × per_step) | step/max_steps (1) ]

  OP-FEATURES (OP_FEATURES_SIZE = 137), from state_extractor.py + config.py:
     op type one-hot ............. NUM_OP_TYPES = 6   (Generic,Matmul,Conv,Pooling,Add,Relu)
     loop upper bounds ........... L = 7              (log2- or max-normalized)
     loop parallel flags ......... L = 7
     load access patterns ........ LS·LSD·L = 2·4·7 = 56   (affine-map coefficients)
     store access patterns ....... LS·LSD·L = 2·4·7 = 56
     arithmetic op counts ........ |ARITH_OPS| = 5    (+, −, ×, ÷, exp)
     ------------------------------------------------- 6+7+7+56+56+5 = 137

  ACTION-HISTORY (history_mode = "success-encoding", default):
     per step: one-hot(action) ⊕ success_flag  =  (total_actions + 1)

  STATE  ── action ──►  apply transform ──►  measure speedup ──►  REWARD
    ▲                                                                │
    └──────────────── next state (updated history) ◄────────────────┘
```

The op features are extracted by dumping the tagged operation's AST (a subprocess to an AST-dumper
binary) and reading loop bounds, parallel annotations, memory-access affine maps, and arithmetic
content. `L=7` is sized for the largest family (`conv_2d_nchw_fchw` has 7 loops); `LSD=4` is the max
indexed-tensor rank (NCHW). The flat-vector design is deliberate — **no GNN**; the policy is an MLP
over this encoding.

### A4.3 Action space (MultiDiscrete) and its integration

`build_action_space(registry, max_n_loops)` (`action_space.py`) constructs a single
`spaces.MultiDiscrete` whose **dimension 0 is the action selector** (over all actions, plus an
implicit `done`) and whose **subsequent dimensions are the concatenated parameter slots** of every
action. A per-action *slot map* records which slots belong to which action; after the selector is
known, `_unpack_action` reads only that action's slots and calls its `decode_params`.

```
[Figure 5] MultiDiscrete action space (example: v48 matmul, 6 actions)

  dim 0  : action selector ∈ {Tiling, LoopInterchange, VecSeq, VecPar,
                              ParTiling, ParThreads, done}
  dims 1+: parameter slots, concatenated per action —
           [ Tiling slots .... | LoopInterchange slot | VecSeq slots | ... ]
              5,5,5,5,5,5,5          (1 enum)           (per-loop)    ...

  one sampled action  →  (selector, that action's slot values)
  e.g. selector=Tiling, slots=[2,1,3,...] → decode → {tile_sizes:[8,4,16,...]}
```

This realizes the intended **hierarchical action**: a macro choice of *which transformation* plus a
micro choice of *its parameters*, learned jointly by one policy.

Alternative `param_mode`s exist (`env_config.py`): `"multidiscrete"` (default, above),
`"two_policy"` (a separate parameter network on a discrete selector), and `"llm"` (an LLM
parametrizer agent). The thesis focuses on `"multidiscrete"`.

### A4.4 Masking: the two operating modes from Layer 3

`masking_mode` (default `"dependencies"`) selects how `action_masks()` constrains the policy:

- **`"dependencies"` — free schedule (denylist).** All actions are available except those blocked by
  `ACTION_DEPENDENCIES` given what already ran (`compute_blocked_indices`). The policy explores
  freely subject only to preconditions, multiplicity caps, and structural blocks.
- **`"schedule_graph"` — graph schedule (allowlist).** Only actions that *extend the successfully
  applied prefix along a `SCHEDULE_GRAPH` path for the kernel's family* are allowed
  (`compute_schedule_allowed`). If a family has no graph, it degrades gracefully to allow-all.
- **`"none"`** — no masking beyond multiplicity caps.

```
[Figure 6] Denylist vs allowlist (v48 matmul, verbatim from registry.py)

 ACTION_DEPENDENCIES  (denylist — "if X ran, Y is now illegal"):
   VectorizationSequential ─blocks─► Tiling, LoopInterchange, VectorizationParallel,
                                     ParallelizationTiling, ParallelizationThreads
   VectorizationParallel   ─blocks─► (same set)
   →  i.e. once you vectorize, nothing structural may follow (lowering is terminal).

 SCHEDULE_GRAPH["matmul"]  (allowlist — branch/decision tree of good shapes):
        ┌─ ParallelizationThreads ─► VectorizationSequential ─► (done)
   root ┤
        ├─ ParallelizationTiling ─► Tiling ─► VectorizationSequential ─► (done)
        │
        └─ ParallelizationTiling ─► LoopInterchange ─► VectorizationParallel ─► (done)

   The policy navigates this tree (choosing the branch) and still tunes every
   action's parameters itself.  Vectorization is always the terminal action.
```

```
[Figure 7] Free vs graph schedule — exploration surface & sample efficiency

  FREE (dependencies):                 GRAPH (schedule_graph):
   action product space, lightly         policy restricted to a few curated,
   pruned by structural blocks           empirically-good skeletons
   ── broad, can discover novel           ── narrow, high sample efficiency
      schedules, slower to learn             learns parameters within known-good
      (more wasted steps)                    shapes; fast convergence
```

This pairing is a core sample-efficiency lever: Layer-3's empirical exploration becomes a learned
prior that shrinks the RL search space (graph mode), while free mode preserves the ability to
discover new schedules.

### A4.5 Reward

Reward is the measured **speedup** of the current/final code over a baseline. Configured via
`env_config.py`:

| Knob | Options (default) | Meaning |
|---|---|---|
| `reward_mode` | `final` / `intermediate` / `schedule` (`final`) | when reward is given and against what. |
| `reward_scale` | `log` / `raw` / `delta` / `relative` (`log`) | transform of the speedup ratio. |
| `reward_baseline` | `mlir` / `torch` (`mlir`) | denominator of the ratio (PyTorch enables "beat-Torch" comparison). |
| penalties | `failed_transform_penalty`, `failed_exec_penalty`, `no_action_penalty` | shape failure handling. |

- **`final`** — reward only at episode end: `log(base_time / final_time)`.
- **`intermediate`** — per-step improvement over the *previous* step (`prev_time / current_time`).
- **`schedule`** — per-step improvement over the *original baseline* (`base_time / current_time`).
- **Scales:** `raw` = ratio; `log` = `log(ratio)` (default, compresses wide dynamic range across
  shapes); `delta` = `1 − 1/ratio` ∈ [0,1); `relative` = `ratio / best_seen` (per-shape normalized).
- **Termination:** an episode ends when the policy selects `done`, when the `tag = "operation_0"` is
  consumed (a terminal lowering action like vectorization), or when `MAX_STEPS` is reached
  (truncation). Baselines are pre-measured and cached in `benchmarks.py`; PyTorch baselines are
  measured on identical shapes for a fair cross-framework speedup.

### A4.6 Execution backends

`executor_type` selects how MLIR is actually run to obtain timings: `local` (in-process MLIR Python
bindings), `dask` (parallel Dask workers), or `slurm` (batch submission). Timings use warm-up +
median timing; the same fixed bufferization + LLVM lowering pipeline is applied before execution
(see A6.3).

---

## A5. PPO training & evaluation

Located in `src/rl/`. Driven by `scripts/train.sh` (training) and `scripts/evaluate.sh` (eval).

| File | Role |
|---|---|
| `train_ppo.py` | Builds env + `MaskablePPO`, callbacks, logging; runs `model.learn`. |
| `evaluate_ppo.py` | Evaluates a trained model (`execution` live re-run or `training-logs`). |
| `behavior_masking.py` | `BehaviorMaskedActorCriticPolicy` (experimental behaviour-only masking). |

### A5.1 Why PPO

PPO is the on-policy, actor-critic, policy-gradient method of choice here because:

- its **clipped surrogate objective** bounds each update, giving stable training without the
  fragility of vanilla policy gradients or the hyperparameter sensitivity of TRPO;
- it reuses each rollout for several epochs of minibatch updates (**sample efficiency** for an
  expensive environment — every step costs a real MLIR compile+run);
- it pairs naturally with **action masking** via the `MaskablePPO` variant (`sb3_contrib`), which is
  essential given the dependency/schedule-graph constraints (A4.4);
- it handles the **MultiDiscrete** (multi-categorical) action distribution out of the box.

### A5.2 Architecture & algorithm

`MaskablePPO` + `ActionMasker` (from `sb3_contrib`) over a stock MLP actor-critic (`MlpPolicy`,
shared backbone, e.g. `[128,128]` or `[256,256,256]` for harder families). The actor head emits a
**multi-categorical** distribution over the MultiDiscrete dims; the critic emits the value estimate.
Advantages via **GAE**. An entropy-coefficient annealing callback decays exploration over training.

```
[Figure 8] PPO training loop with the compiler in the loop

  ┌──────────────── rollout (n_steps) ────────────────┐
  │ obs ─► policy.predict(obs, action_masks) ─► action │
  │ env.step(action):                                  │
  │     apply Transform-dialect action  ──┐            │
  │     bufferize + lower + EXECUTE MLIR  ─┴► time      │ ← real compiler work
  │     reward = f(speedup)                            │
  └───────────────────────┬────────────────────────────┘
                          ▼
        GAE advantages (γ, λ)  +  returns
                          ▼
   n_epochs × minibatches:  clipped PPO loss + value loss − entropy bonus
                          ▼
        gradient step (clip grad norm) ─► updated policy
                          ▼
      periodic FullEvalCallback (greedy + sampled) ─► best_model.zip
```

### A5.3 Curated, thesis-relevant configuration

The conceptually important knobs (defaults in `train_ppo.py` / `env_config.py`; full set persisted
per run to `config.json`):

| Group | Knob | Default | Note |
|---|---|---|---|
| PPO core | `lr`, `n_steps`, `batch_size`, `n_epochs` | 3e-4, 128, 32, 8 | standard PPO. |
| PPO core | `gamma`, `gae_lambda`, `clip_range` | 0.99, 0.95, 0.2 | discount / GAE / clip. |
| PPO core | `ent_coef → ent_coef_final` | 5e-3 → 5e-5 | annealed (linear/exp). |
| PPO core | `vf_coef`, `max_grad_norm` | 0.05, 0.5 | value weight / grad clip. |
| Net | `net_arch` | `[128,128]` | larger for conv/ml. |
| Env | `action_version` | `v10` (per experiment) | which generated action set. |
| Env | `param_mode` | `multidiscrete` | hierarchical action. |
| Env | `masking_mode` | `dependencies` | free vs `schedule_graph`. |
| Env | `reward_mode` / `reward_scale` / `reward_baseline` | `final` / `log` / `mlir` | reward design. |
| Env | `max_steps` | 7 | episode length. |
| Exec | `executor_type` | `dask` | `local` / `dask` / `slurm`. |

### A5.4 Evaluation & logging

- **`evaluate_ppo.py` modes:** `execution` re-runs the best model greedily (and optionally with
  stochastic sampling), measuring real speedups and emitting per-kernel/summary CSVs + `results.json`;
  `training-logs` reconstructs the best evaluation from the training-time JSON logs without
  re-execution (fast, reproducible).
- **Logging stack:** Weights & Biases (episode reward/length/speedup, per-action success rates,
  convergence indicators) + the SB3 logger (`stdout`, `csv`, `tensorboard`) + custom JSON eval logs
  under `per_benchmark_evals/`.

---

## A6. The MLIR-Torch MCP (the agent's grounding layer)

The MCP server is how the LLM agents *act on* and *measure* MLIR and PyTorch. Located in `src/mcp/`,
documented in `docs/MCP.md` and `docs/MCP_MINIMAL.md`.

```
[Figure 9] MCP as the bridge

   Claude agent (Layer 2 / Layer 3)
        │  MCP tool calls
        ▼
   FastMCP server  "mlir-tools"   (src/mcp/mcp_server.py | _minimal.py)
        │
        ├─ transform_mlir_code(code, transform_ir) ─► transformed MLIR
        ├─ execute_mlir_code(code)               ─► (time_ms, success)   ─┐
        ├─ execute_torch_{matmul,conv2d,add,        PyTorch baselines  ─┤ via SLURM
        │     pooling_nchw_max,relu}_by_shape(...)  on identical shapes ─┘ submit+poll
        ├─ measure_speedup(base, opt[, torch])   ─► speedup ratios
        └─ delegate_documentation_lookup(task)   ─► Transform-dialect docs

   Per-action server  "rl-action-v<x>"  (src/actions/v<x>/mcp.py)
        └─ one tool per action: (code, params) -> (pre, transformed, post)
```

### A6.1 Tool catalog & roles

- **`transform_mlir_code` / `execute_mlir_code`** — apply Transform IR and run a payload (median
  timing). Used by Layer 2's 5-step validation.
- **`execute_torch_*_by_shape`** — measure PyTorch on the same shape, giving a fair cross-framework
  baseline (`reward_baseline = "torch"` and Layer-3 "beat-PyTorch" tracking).
- **`measure_speedup`** — `base/opt` and optional `torch/opt` ratios.
- **`delegate_documentation_lookup`** — a deterministic retrieval agent over the Transform-dialect
  docs, so the synthesizing agent grounds op names/handles instead of hallucinating them.

### A6.2 Full vs minimal server

The **full** server (`mcp_server.py`) exposes transformation + documentation lookup and is used
during **Layer 2** synthesis. The **minimal** server (`mcp_server_minimal.py`) drops those and is
used during **Layer 3**, where transformations must go exclusively through the per-action
`rl-action-v<x>` tools (the agent is forbidden from writing raw Transform IR).

### A6.3 Execution model & default pipeline

Tools submit MLIR/PyTorch jobs to SLURM (`src/mcp/utils.py`: submit → poll `squeue` with
`SLURM_TIMEOUT=300s` → parse JSON result) or run via the in-process executors. A standard
bufferization + lowering Transform sequence and a fixed LLVM **pass pipeline**
(`src/utils/transformation.py`) are applied before execution (LICM, empty-tensor elimination,
one-shot bufferization, vector lowering, → LLVM, OpenMP for parallel loops, etc.), so every measured
schedule is lowered identically.

---

## A7. Cross-cutting concerns

- **Versioning (`v0 … v52`).** Each pipeline run produces an independent action version; the
  benchmark family and the layer parameters (e.g. intents/transformations counts) are recorded in
  the run. Recent versions pair a family with a schedule graph (e.g. `v48` matmul, `v49` conv2d,
  `v50` pooling, `v51` add, `v52` relu).
- **Benchmark families.** `data/benchmarks/` holds `dataset_matmul`, `dataset_conv2d`,
  `dataset_conv2d_img2col`, `dataset_pooling`, `dataset_add`, `dataset_relu`, `dataset_ml`, plus
  `paper_*` and `standard` sets, each split into `train`/`eval`, every kernel pre-tagged.
- **Free vs graph schedule & sample efficiency.** See A4.4 / Figure 7 — the central knob trading
  exploration breadth for convergence speed.
- **Hardware target.** Fixed Broadwell/AVX2/28-core assumptions (A1.4) make the agents' reasoning
  and the divisor/vector-size constraints concrete.
- **Reproducibility.** Every training run persists `config.json`, a seed, W&B run id, checkpoints,
  and `best_model.zip`; evaluation can be replayed from logs without re-execution.

## A8. Consolidated file / artifact map

| Component | Key files | Role |
|---|---|---|
| Pipeline drivers | `scripts/claude_{enumeration,implementation,exploration}.sh` | SLURM jobs that run the Claude agent per layer. |
| Prompt generators | `src/prompts/claude_{enumeration,implementation,exploration}.py` | Build the agent prompt + embed benchmark representation. |
| Methodology specs | `resources/prompts/v1/{action_enumeration,action_implementation,schedule_exploration,system_description}.md` | The agents' system prompts (the methodology). |
| Benchmark representation | `src/data/benchmarks.py` (`format_for_prompt`, `group_by_family`, `load_benchmark_set`) | Kernel loading + prompt formatting. |
| Action contract | `src/actions/base.py`; template `src/actions/v0/*` | `ActionBase` interface + empty reference skeleton. |
| Action examples | `src/actions/v10/*`, `src/actions/v48/*` | Concrete generated actions + registry/deps/graph. |
| Action testing | `src/actions/test.py` | `test_action` standalone harness. |
| RL environment | `src/env/{mlir_opt_env,action_space,action_registry,state_extractor,env_config,benchmarks}.py` | The MDP, action space, masking, state, config. |
| PPO training/eval | `src/rl/{train_ppo,evaluate_ppo,behavior_masking}.py`; `scripts/{train,evaluate}.sh` | MaskablePPO training & evaluation. |
| MCP | `src/mcp/{mcp_server,mcp_server_minimal,utils}.py`; `docs/MCP.md`, `docs/MCP_MINIMAL.md` | Agent grounding: transform/execute/baseline/docs. |
| Lowering pipeline | `src/utils/transformation.py` | `run_transform_code`, default bufferize/lower + LLVM passes. |
| Global config | `src/config.py` | Constants (`L=7`, `OP_FEATURES_SIZE=137`, `MAX_PARAM_SLOTS=7`, `N_CORES=28`, timeouts, paths). |

---

# PART B — Proposed "Design and Implementation" Chapter Outline

> **How to read Part B.** Each subsection lists: **Purpose** (what the section achieves for the
> reader), **Cover** (the content, in research register), **Draws on** (Part A sections + figures),
> and **Claim** (the argument to land). Numbering is indicative (assume this is Chapter 3); adapt to
> the university template. The recommended narrative arc is summarized at the end.

### 3.1 Motivation and problem statement
- **Purpose:** establish *why* this work exists before any machinery is introduced.
- **Cover:** RL for compiler optimization in one paragraph; then the real bottleneck — manual
  action-space design is expensive, brittle, and slow (legality, composition, parameterization).
  Frame the three sub-pains (time-consuming engineering, error-prone Transform-dialect work, painful
  experimentation across kernel families).
- **Draws on:** A1.1.
- **Claim:** *the action space, not the policy, is the limiting factor — and it can be automated.*

### 3.2 Design overview and principles
- **Purpose:** give the reader the whole picture on one page before drilling in.
- **Cover:** the three integrated worlds (Compiler / LLM Agent / RL) and the closed loop; the five
  design principles (action≠script, separation of concerns, composability-first, RL-friendliness,
  generalizability); the target assumptions (Broadwell/AVX2/28-core) as design context.
- **Draws on:** A1.2–A1.4; **Figure 1**.
- **Claim:** *the contribution is a single integrated system whose principles make the parts fit.*

### 3.3 Agentic foundation
- **Purpose:** justify the agent and explain how it is grounded in the compiler.
- **Cover:** choice of Claude as the coding agent (motivate with coding-benchmark standing and the
  agentic Claude Code CLI execution model); the role of versioned methodology prompts; **the
  MLIR-Torch MCP** as the grounding layer that lets the agent *transform, execute, and benchmark*
  MLIR and PyTorch (tools, SLURM execution, full vs minimal server, documentation retrieval to
  prevent hallucinated dialect usage).
- **Draws on:** A6 (and A2's invocation pattern); **Figure 9**.
- **Claim:** *a strong coding agent, grounded in real compiler feedback, can do compiler engineering.*

### 3.4 The automatic action-generation pipeline
- **Purpose:** the core contribution — present the three layers and the separation of concerns.
- **Cover:** open with the separation-of-concerns thesis (what / how / does-it-compose). Then one
  subsection per layer, each stating motivation, agent role, process, **inputs and outputs**, and
  LLM/MCP usage:
  - **3.4.1 Layer 1 — Enumeration** (A2.1): abstract intents → macro actions + `action_template`;
    granularity rule; metadata output.
  - **3.4.2 Layer 2 — Implementation** (A2.2): one transformation → one executable `ActionBase`
    action; the 5-step MCP self-validation; the tagging, multiplicity, and safety contracts.
  - **3.4.3 Layer 3 — Schedule Exploration** (A2.3): phased composability discovery; shapes-not-
    parameters scoping; parallel sub-agents; the `SCHEDULE_GRAPH` / `ACTION_DEPENDENCIES` outputs.
  - **3.4.4 The action contract as the unifying interface** (A3): `ActionBase`, lifecycle, tagging
    categories, the Tiling worked example.
- **Draws on:** A2, A3; **Figures 2 and 3**; the v48 example (A3.6 / A4.4).
- **Claim:** *separation of concerns is what makes automated action synthesis robust and auditable.*

### 3.5 From generated actions to an RL action space
- **Purpose:** show the seamless hand-off from pipeline output to a learnable action space.
- **Cover:** the registry loading a version; the **MultiDiscrete** mapping (selector + per-action
  parameter slots, slot map, `decode_params`); legality masking — `valid_param_mask` (divisibility),
  the dependency denylist, and the schedule-graph allowlist; why this is RL-friendly (discrete macro,
  parameterized micro, maskable).
- **Draws on:** A3.4, A4.3, A4.4; **Figures 5 and 6**.
- **Claim:** *the generated contract maps directly onto a hierarchical, maskable RL action space — no
  manual glue.*

### 3.6 RL formulation
- **Purpose:** define the MDP precisely.
- **Cover:** state/observation (loop-nest features + action history + progress; flat-vector, MLP, no
  GNN — and why); action (as in 3.5); reward design (final / intermediate / schedule; log/raw/delta/
  relative scales; MLIR vs PyTorch baseline; penalties; tag-consumption termination); **free vs graph
  schedule modes** and their sample-efficiency trade-off.
- **Draws on:** A4.2, A4.5, A4.4; **Figures 4 and 7**.
- **Claim:** *the MDP is shaped so that empirical schedule knowledge (Layer 3) becomes a learning
  prior, improving sample efficiency.*

### 3.7 Learning algorithm
- **Purpose:** justify and describe the optimizer.
- **Cover:** why PPO (clipped surrogate stability, on-policy sample reuse for an expensive env,
  multi-categorical support, natural fit with masking); the `MaskablePPO` + `ActionMasker`
  realization; actor-critic MLP, GAE, entropy annealing; the training loop with the compiler in the
  loop; the curated hyperparameters; dual-mode evaluation and logging.
- **Draws on:** A5; **Figure 8**.
- **Claim:** *MaskablePPO is the right algorithm for a masked, MultiDiscrete, expensive-step
  environment.*

### 3.8 Implementation details and configuration
- **Purpose:** the engineering substrate, kept brief and reference-like.
- **Cover:** execution backends (local/dask/slurm) and the fixed lowering pipeline; benchmark
  families and tagging; the `v<x>` versioning discipline; reproducibility (config.json, seeds,
  checkpoints).
- **Draws on:** A4.6, A6.3, A7.
- **Claim:** *the system is reproducible and built to iterate fast across families and action sets.*

### 3.9 Synthesis: closing the loop
- **Purpose:** tie the chapter together and bridge to Evaluation.
- **Cover:** restate the loop — agent-*generated* actions, empirically *validated* schedules,
  RL-*consumed* action space, all grounded in *measured* speedups; note what this enables that manual
  design did not (rapid per-family action sets, auditable contracts); forward-reference the
  Evaluation chapter (speedups vs MLIR baseline and PyTorch).
- **Draws on:** A1.2, A7; **Figure 1** (reprise).
- **Claim:** *the contribution is an end-to-end, automated, reproducible replacement for manual
  compiler RL action-space engineering.*

---

## Recommended narrative arc (the story spine)

> **Pain → Idea → Agent → Pipeline → Action space → MDP → Learner → Loop.**
>
> "Compiler RL is bottlenecked by its action space (3.1). We integrate three worlds to automate it
> (3.2). A strong coding agent, grounded in the compiler via MCP, does the engineering (3.3). It runs
> a three-layer pipeline — enumerate, implement, explore — separated by concern (3.4). The generated,
> contract-bearing actions map directly onto a maskable MultiDiscrete RL action space (3.5), inside a
> carefully shaped MDP (3.6), optimized by MaskablePPO (3.7), on a reproducible substrate (3.8),
> closing a fully automated loop (3.9)."

Keep the *agentic synthesis* (3.3–3.4) as the chapter's centre of gravity — it is the novel
contribution; the RL machinery (3.5–3.7), while substantial, is the *consumer* of that contribution
and should be framed as "the action space the agent built, made learnable."

## Figure map (proposed thesis figures → Part A diagrams)

| Thesis figure | Part A source | Shows |
|---|---|---|
| End-to-end system architecture | Figure 1 (A1.2) | Compiler ⟷ LLM Agent ⟷ RL closed loop. |
| Three-layer pipeline with artifacts | Figure 2 (A2) | Enumeration → Implementation → Exploration + outputs. |
| Action contract lifecycle | Figure 3 (A3.1) | precondition → preprocess → implement → postcondition. |
| RL MDP & observation vector | Figure 4 (A4.2) | state/action/reward + the 137-d feature breakdown. |
| MultiDiscrete action space | Figure 5 (A4.3) | selector + per-action parameter slots. |
| Denylist vs allowlist (v48) | Figure 6 (A4.4) | `ACTION_DEPENDENCIES` graph & `SCHEDULE_GRAPH` tree. |
| Free vs graph schedule | Figure 7 (A4.4) | exploration breadth vs sample efficiency. |
| PPO training loop | Figure 8 (A5.2) | rollout → GAE → clipped update, compiler-in-the-loop. |
| MCP as the bridge | Figure 9 (A6) | tool catalog and grounding role. |
