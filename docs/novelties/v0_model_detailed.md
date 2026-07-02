# RL Agent Model Architecture (`HiearchyModel`) — Detailed Breakdown

This document explains the neural-network architecture of the baseline RL autoscheduler in **plain, step-by-step terms**. It covers every tensor shape, the rationale behind each layer, concrete numerical examples, and the known limitations of this design.

> **Scope**: `rl_autoschedular/model.py` and its interaction with `observation.py` / `actions/`.  
> **What it is NOT**: training algorithm (PPO), environment dynamics, or the MLIR execution engine.

---

## 1. High-Level Picture

The model is a single PyTorch `nn.Module` named `HiearchyModel` (note the typo in the source code). It contains two sub-models that share the **same** embedding backbone:

```text
Observation (batch_size, obs_dim)
        │
        ▼
   LSTMEmbedding  ──► (batch_size, 411 + history_size)
        │
   ┌────┴────┐
   ▼         ▼
PolicyModel   ValueModel
   │           │
   ▼           ▼
[action distributions]   [state value scalar]
```

* **PolicyModel** — decides *which* transformation to apply and *with what parameters*.
* **ValueModel** — estimates how good the current state is (used only for PPO advantage computation).

Both models are trained, but only the **PolicyModel** drives the agent at inference time.

---

## 2. Input: The Observation Tensor

Before the neural network sees anything, the raw `OperationState` is flattened into a single 1-D vector by `Observation.from_state()`. The full observation is a concatenation of five parts:

| Part | What it encodes | Size |
|---|---|---|
| **OpFeatures** | One-hot op type (5), loop upper bounds (`L`), iterator types (`L`), flattened load/store access matrices (`LS × LSD × L` each), arithmetic op counts (5) | `5 + 2L + 2·LS·LSD·L + 5` |
| **ProducerOpFeatures** | Identical structure, but for the *producer* op; all zeros if there is no producer | same as above |
| **ActionHistory** | Flattened one-hot record of every past action taken on the *current* operation | `ActionSpace.cumulative_history_sizes()[-1]` |
| **NumLoops** | Scalar: how many nested loops the current op actually has | 1 |
| **ActionMask** | Boolean mask that tells the network which actions / parameters are legal right now | `ActionSpace.cumulative_mask_sizes()[-1]` |

> **Config knobs** that control these sizes:
> * `max_num_loops` (`L`) — default is usually 7 or 8.
> * `max_num_stores_loads` (`LS`) — caps load/store tensors.
> * `max_num_load_store_dim` (`LSD`) — caps dimensions per load/store.
> * `num_tile_sizes` — how many power-of-2 tile sizes the network can emit.
> * `truncate` — max actions per operation (episode length cap).

### 2.1 Concrete Example: OpFeatures for a 3-loop matmul

Assume `L = 7`, `LS = 4`, `LSD = 3`.

```
OperationType one-hot:  [0, 1, 0, 0, 0]          # matmul
Loop upper bounds:      [256, 256, 256, 0, 0, 0, 0]   # padded to L
type=parallel flags:    [1, 1, 0, 0, 0, 0, 0]
Load matrices (flatten): 4 × 3 × 7 = 84 floats
Store matrices (flatten): 4 × 3 × 7 = 84 floats
Arithmetic counts:      [+, -, *, /, exp]  → 5 floats
─────────────────────────────────────────────────────
Total OpFeatures size = 5 + 7 + 7 + 84 + 84 + 5 = 192
```

*Bounds are normalized*:
* `normalize_bounds = 'log'` → `log2(256) = 8.0`
* `normalize_bounds = 'max'` → `256 / 4096 = 0.0625`

---

## 3. LSTMEmbedding — The Shared Backbone

This module turns the raw observation into a fixed-size vector that both policy and value networks consume.

### 3.1 Architecture

```python
LSTMEmbedding(
  (embedding): Sequential(
    (0): Linear(OpFeatures.size(), 512)
    (1): ELU()
    (2): Dropout(p=0.225)
    (3): Linear(512, 512)
    (4): ELU()
    (5): Dropout(p=0.225)
  )
  (lstm): LSTM(512, 411, num_layers=1)
)
```

### 3.2 Why an LSTM over two ops?

The observation contains **both** the current (consumer) op and its producer op. The network needs to understand the *relationship* between them — for example, whether fusing the producer into the consumer is worthwhile.

```text
Step 1: consumer_feats  ──► embedding MLP ──► (512) ──┐
                                                        ├──► LSTM ──► hidden (411)
Step 2: producer_feats  ──► embedding MLP ──► (512) ──┘
```

* The same MLP embeds both op feature vectors independently.
* They are fed as a **2-step sequence** into a unidirectional LSTM with hidden size 411.
* Only the **final hidden state** (`hn`) is kept; the cell state and all intermediate outputs are discarded.

### 3.3 Output size

```python
output_size = 411 + ActionHistory.size()
```

The final hidden state (411) is concatenated with the raw `ActionHistory` vector. The history is **not** processed by the MLP or LSTM; it is appended verbatim. This means the policy/value heads receive:

1. A learned, compressed representation of the consumer+producer features.
2. An explicit, uncompressed memory of everything the agent has already done to this operation.

### 3.4 Numerical Example (batch_size = 1)

```
consumer_feats  shape: (1, 192)
producer_feats  shape: (1, 192)      # or 192 zeros if no producer

consumer_embeddings = MLP(consumer_feats)   → (1, 512)
producer_embeddings = MLP(producer_feats)   → (1, 512)

stacked = torch.cat([consumer, producer])    → (2, 512)
_, (hn, _) = LSTM(stacked)                   → hn is (1, 1, 411)
hn.squeeze(0)                                → (1, 411)

action_history                               → (1, history_size)
final_embedding = cat([hn, history])         → (1, 411 + history_size)
```

---

## 4. PolicyModel — Deciding What To Do

### 4.1 Backbone

```python
backbone = Sequential(
    Linear(411 + history_size, 512), ReLU(),
    Linear(512, 512),             ReLU(),
    Linear(512, 512),             ReLU()
)
```

### 4.2 Multi-Head Output

After the shared backbone, there is **not** a single output layer. There are *separate heads* for:

1. **Head 0** — action *selection* logits. Size = `ActionSpace.size()` = **6** (one per action type).
2. **Heads 1..N** — parameter logits for each action type. Size = `action.network_output_size()`.

```python
output_sizes = [
    6,                       # NoTransformation, Tiling, TP, TPF, Interchange, Vectorization
    Tiling.network_output_size(),
    TiledParallelization.network_output_size(),
    TiledFusion.network_output_size(),
    Interchange.network_output_size(),
    Vectorization.network_output_size(),
]
```

Some actions have **no parameters** (e.g. `NoTransformation`, `Vectorization`). Their head is `None`.

### 4.3 Concrete Head Sizes (typical config)

With `L = 7`, `num_tile_sizes = 6`, `interchange_mode = 'enumerate'`:

| Action | `network_output_size()` | Meaning |
|---|---|---|
| NoTransformation | 0 | terminal, no params |
| Tiling | `7 × (6 + 1) = 49` | per-loop categorical over 7 tile choices (0 = no tile, 1..6 = power-of-2 sizes) |
| TiledParallelization | 49 | inherits Tiling params |
| TiledFusion | 49 | inherits Tiling params |
| Interchange | `3·7 − 6 = 15` | index into pre-computed 1c/2c/3c swap candidates |
| Vectorization | 0 | terminal, no params |

> If `interchange_mode = 'pointers'`, the size becomes `L = 7`.  
> If `interchange_mode = 'continuous'`, the size becomes `1` (a single Normal distribution).

### 4.4 Action Masking — Making Illegal Choices Impossible

The policy does not simply take `softmax` over all logits. It uses **boolean masks** extracted from the observation to set invalid logits to `-inf` *before* the `Categorical` distribution is built.

```python
# Pseudo-code of what ActionSpace.distributions() does
selection_logits = head0(embedded)
selection_mask   = obs_mask[:, :6]
selection_dist   = Categorical(logits=selection_logits.where(selection_mask, -inf))

for each action i with parameters:
    param_logits = head_i(embedded)
    param_mask   = obs_mask[:, cum_mask[i] : cum_mask[i+1]]
    param_dist   = action.distribution(param_logits.where(param_mask, -inf))
```

This guarantees the agent **can never sample**:
* A fusion action when there is no producer.
* A tile size that does not divide the loop bound.
* An interchange candidate beyond the number of loops.

---

## 5. ValueModel — Estimating State Quality

The value network is almost identical to the policy backbone, but it ends in a single scalar:

```python
ValueModel(
  (lstm): LSTMEmbedding()
  (network): Sequential(
    Linear(lstm.output_size, 512), ReLU(),
    Linear(512, 512),             ReLU(),
    Linear(512, 512),             ReLU(),
    Linear(512, 1)
  )
)
```

* Input: same observation tensor.
* Output: one float per batch item, squeezed to shape `(batch_size,)`.
* Loss: MSE against PPO returns. Optional clipping (`value_clip = True`):
  ```python
  v_clipped = values + clamp(new_values - values, -0.2, 0.2)
  loss = max((returns - v_clipped)^2, (returns - new_values)^2).mean()
  ```

---

## 6. How Actions Are Represented: The Hierarchical Index

A single sampled "action" is not one number. It is a **flat vector** called `actions_index`:

```
index[0]      → which action type was chosen   (0..5)
index[1:1+T]  → parameters for Tiling
index[1+T:...]→ parameters for TP, TPF, Interchange, ...
```

The exact slice positions are computed by `ActionSpace.cumulative_params_sizes()`.

### 6.1 Example: Tiling a 3-loop op

Config: `max_num_loops = 7`, `num_tile_sizes = 6`.

```
ActionSpace.cumulative_params_sizes() = [1, 8, 15, 22, 23, 23]
# index 0       → action type
# index 1..7    → Tiling params (one per loop, padded to 7)
# index 8..14   → TP params
# index 15..21  → TPF params
# index 22      → Interchange param
# index 23      → (no params for Vectorization)
```

If the agent decides to **tile** with parameters `[32, 0, 64]`:

```
index = [1,            # action type = Tiling
         6, 0, 7, 0, 0, 0, 0,   # loop 0→32 (2^5, index 6), loop 1→0, loop 2→64 (2^6, index 7)
         0, 0, 0, 0, 0, 0, 0,   # TP slots (ignored because action type ≠ TP)
         ...]
```

The `ActionSpace` methods know which slice to look at based on the first element (`index[0]`). Parameters belonging to *unselected* action types are ignored during log-probability computation.

---

## 7. Sampling — From Logits to an Action

### 7.1 `model.sample(obs, greedy=False, eps=None)`

```python
distributions    = policy_model(obs)          # list of Distribution objects
eps_distributions= ActionSpace.uniform_distributions(obs)  # same structure, uniform over allowed choices

# 1. Decide: explore uniformly?
uniform = (eps is not None) and (random() < eps)

# 2. Sample action type
if greedy:      action_type = argmax(probs)
elif uniform:   action_type = uniform_dist.sample()
else:           action_type = categorical_dist.sample()

# 3. Sample parameters for *all* action types (only the chosen type matters later)
for each param distribution:
    sample parameter indices

# 4. Compute log-probability and entropy
#    Only the log-prob of the *selected* action type + its parameters is kept.
```

### 7.2 Epsilon Exploration

When `eps` is set (e.g. `eps = 0.1`), there is a 10% chance per step to sample from a **uniform distribution over the legal actions** instead of the learned policy. This is classic epsilon-greedy, but constrained by the action mask so the agent never samples illegal moves even while exploring.

The log-probability used for PPO is a mixture:

```python
log_p = log( (1 - eps) * exp(policy_log_p) + eps * exp(uniform_log_p) )
```

This ensures the policy gradient is still well-defined even during exploratory steps.

---

## 8. Detailed Example: Forward Pass for a Single State

Let's trace a concrete forward pass with made-up but realistic numbers.

**Config**: `L=7`, `LS=4`, `LSD=3`, `num_tile_sizes=6`, `truncate=10`, `interchange_mode='enumerate'`.

**State**: A `matmul` with 3 loops, no producer, no history yet.

### Step 1 — Build Observation

```
OpFeatures size           = 5 + 2·7 + 2·4·3·7 + 5 = 192
ProducerOpFeatures size   = 192  (all zeros)
ActionHistory size        = truncate × (Tiling.history_size + Interchange.history_size + ...)
                          = 10 × (7·7 + 7·7 + ...)  ≈  large, say 980
NumLoops size             = 1
ActionMask size           = 6 + 49 + 49 + 49 + 15 + 0 = 168
─────────────────────────────────────────────────────────────
Total observation size    ≈ 192 + 192 + 980 + 1 + 168 = 1533
```

### Step 2 — LSTMEmbedding

```
consumer_feats  (1, 192) ──► MLP ──► (1, 512)
producer_feats  (1, 192) ──► MLP ──► (1, 512)

stack (2, 512) ──► LSTM(512, 411) ──► final_hidden (1, 411)

cat([411, history(1, 980)]) ──► (1, 1391)
```

### Step 3 — PolicyModel

```
backbone(1391) ──► (1, 512)

head0: Linear(512, 6)     ──► logits for [NT, T, TP, TPF, I, V]
head1: Linear(512, 49)    ──► Tiling params (7 loops × 7 choices)
head2: Linear(512, 49)    ──► TP params
head3: Linear(512, 49)    ──► TPF params
head4: Linear(512, 15)    ──► Interchange candidates
head5: None               ──► Vectorization has no params
```

Mask is applied; `Categorical` / `Normal` distributions are built. Sampling yields:

```
actions_index = [1, 6, 0, 7, 0, 0, 0, 0, 0, ...]   # Tiling, loop0=32, loop1=none, loop2=64
actions_log_p = -2.34                               # joint log-prob of Tiling + those params
entropy       = 4.12                                # total entropy of all distributions
```

### Step 4 — ValueModel

```
lstm(obs) ──► (1, 1391) ──► network ──► (1, 1) ──► squeeze ──► 0.87
```

The scalar `0.87` is the predicted value of the state before any reward is observed.

---

## 9. Limitations of This Baseline Model

### 9.1 Fixed Observation Size

All tensors are padded to `max_num_loops`, `max_num_stores_loads`, etc. A benchmark with only 2 loops still produces a vector padded for 7. This wastes computation and parameters, though it simplifies batching.

### 9.2 LSTM Only Sees Two Ops

The LSTM is hard-coded to consume exactly two steps: **consumer** then **producer**. If a benchmark has complex multi-consumer or multi-producer patterns, the model never sees the full graph topology — only one local edge at a time.

### 9.3 No Graph-Level Message Passing

There is no GNN, no attention over the full DAG, and no global benchmark embedding. The model cannot learn patterns like "if op A is tiled by 32, then op B should also be tiled by 32" unless that correlation happens to be captured implicitly in the shared LSTM weights across steps.

### 9.4 Action History Is Appended Raw

The `ActionHistory` is concatenated *after* the LSTM, without any additional processing. For long sequences (e.g. `truncate = 20`), the history vector can dominate the input dimension, yet it receives no recurrent compression. This makes the model sensitive to history length.

### 9.5 Separate Policy / Value Networks with Shared LSTM Only

While the LSTM is shared, the **backbones** (the 3-layer MLPs) are **not** shared between policy and value. This doubles the parameter count and can cause the two networks to diverge in how they interpret the embedding.

### 9.6 Tiling Parameterization Is Power-of-2 Only

The network does not output arbitrary tile sizes. It outputs an index that maps to `2^(param-1)`. This is efficient but eliminates non-power-of-2 tilings that might be optimal on real hardware.

### 9.7 Interchange Mode Determines Architecture

Switching `interchange_mode` from `enumerate` to `pointers` or `continuous` changes the size of Head 4. This means the **network shape is config-dependent**; you cannot load a checkpoint trained with `enumerate` into a model instantiated with `pointers`.

### 9.8 No Hardware Features in the State

The observation contains nothing about the target CPU (cache sizes, vector width, number of cores). The agent must learn hardware-specific optima indirectly through execution-time rewards, which is sample-inefficient.

---

## 10. Summary Cheat-Sheet

| Component | Input Shape | Output Shape | Key Hyperparameters |
|---|---|---|---|
| `OpFeatures` | raw state | `(OpFeatures.size(),)` | `max_num_loops`, `max_num_stores_loads`, `max_num_load_store_dim` |
| `LSTMEmbedding` | `(B, obs_dim)` | `(B, 411 + history_size)` | `embedding_size = 411`, dropout = 0.225 |
| `PolicyModel.backbone` | `(B, 411 + hist)` | `(B, 512)` | ReLU activation |
| `PolicyModel.heads[0]` | `(B, 512)` | `(B, 6)` | action selection logits |
| `PolicyModel.heads[i]` | `(B, 512)` | `(B, action_i.output_size)` | parameter logits per action type |
| `ValueModel.network` | `(B, 411 + hist)` | `(B, 1)` | MSE loss, optional clipping |
| `actions_index` | — | `(B, 1 + max_params)` | flat hierarchical index |

---

## 11. Where to Read the Code

| File | What to look at |
|---|---|
| `rl_autoschedular/model.py` | `HiearchyModel`, `LSTMEmbedding`, `PolicyModel`, `ValueModel` |
| `rl_autoschedular/observation.py` | `Observation`, `OpFeatures`, `ActionHistory`, `ActionMask` |
| `rl_autoschedular/actions/__init__.py` | `ActionSpace.distributions()`, `sample()`, `distributions_stats()` |
| `rl_autoschedular/actions/tiling.py` | Concrete example of param encoding, masking, and history |
| `rl_autoschedular/actions/interchange.py` | Multi-step (`ready=False`) and continuous (`Normal`) action patterns |

---

*End of document.*
