# V3: Transformer Loop-Nest Encoder

## Overview

Version 3 replaces the previous LSTM embedding with a transformer-based encoder that models loop-nest structure using attention. The objective is to improve representation quality for complex nested loops while keeping the rest of the RL pipeline unchanged.

This version is intentionally scoped to one novelty only:
- Encoder architecture upgrade from LSTM to Transformer

No reward, action-space, PPO, or environment behavior changes are introduced in V3.

## Why this change

The baseline embedding path processes consumer and producer operation features with an LSTM. While simple and efficient, recurrent processing can under-represent long-range interactions between loop levels and producer-consumer contexts.

The transformer encoder provides:
- Parallel processing of loop-level tokens
- Attention-based interaction between outer and inner loops
- Explicit structural conditioning through role/type/depth embeddings

## Design goals

1. Keep V3 isolated to one novelty (encoder only).
2. Preserve existing training and evaluation interfaces.
3. Keep policy/value heads unchanged.
4. Make architecture tunable from config.

## Implemented architecture

## 1) Structured tokenization

Implemented in `rl_autoschedular_v3/model.py` inside `TransformerEmbedding`.

For each observation sample, the encoder constructs a token sequence:
1. CLS token
2. Consumer summary token (projection of full consumer op feature vector)
3. Producer summary token (projection of full producer op feature vector)
4. Consumer per-loop tokens (one token per loop level)
5. Producer per-loop tokens (one token per loop level)
6. Optional action-history token (configurable)

Per-loop token inputs are built from:
- loop upper bound
- loop iterator type
- loop-specific load access coefficients
- loop-specific store access coefficients
- operation-type one-hot context
- arithmetic-op count context

This allows each loop token to carry both local loop details and global operation context.

## 2) Structural embeddings

Each token receives additive structural embeddings:
- Role embedding: global / consumer / producer
- Token-type embedding: cls / summary / loop / action-history
- Depth embedding: loop depth (0 for non-loop tokens, 1..L for loop tokens)

These embeddings provide positional hierarchy and semantic context without changing observation format.

## 3) Transformer encoder core

Encoder stack:
- `nn.TransformerEncoder`
- pre-norm (`norm_first=True`)
- configurable heads/layers/hidden dimensions
- configurable feed-forward activation (`relu` or `gelu`)

Token validity mask:
- Consumer loop tokens are masked using `NumLoops`.
- Producer loop tokens are masked using non-zero loop signal detection.
- Mask is passed as `src_key_padding_mask` to avoid learning from padded loops.

## 4) Output pooling and compatibility

Pooling options:
- `cls`: use CLS token output
- `mean`: masked mean over valid tokens

Compatibility contract:
- If action-history token is disabled, pooled transformer output is concatenated with raw ActionHistory, preserving a baseline-like interface.
- If enabled, action history is fully represented in-token and output remains transformer pooled only.

Policy and value heads still consume a single embedding vector and require no architectural rewrite.

## Implementation details by file

- `rl_autoschedular_v3/model.py`
  - Replaced `LSTMEmbedding` with `TransformerEmbedding`.
  - Updated `PolicyModel` and `ValueModel` to use transformer embedding path.
  - Added feature-splitting utilities for loop-level token construction.
  - Added structural embeddings and pooling logic.

- `utils/config.py`
  - Added transformer architecture fields:
    - `transformer_d_model`
    - `transformer_nhead`
    - `transformer_num_layers`
    - `transformer_ffn_dim`
    - `transformer_dropout`
    - `transformer_activation`
    - `transformer_pooling`
    - `transformer_use_action_history_token`

- `config/example.json`
  - Added default transformer field values.

- `README.md`
  - Added documentation for transformer config fields.

## Configuration

Example V3 config:

```json
{
  "implementation": "rl_autoschedular_v3",
  "transformer_d_model": 256,
  "transformer_nhead": 8,
  "transformer_num_layers": 3,
  "transformer_ffn_dim": 1024,
  "transformer_dropout": 0.1,
  "transformer_activation": "gelu",
  "transformer_pooling": "cls",
  "transformer_use_action_history_token": false
}
```

Notes:
- Keep `transformer_d_model % transformer_nhead == 0`.
- Start with `cls` pooling and action-history token disabled for stable baselines.

## How to use

1. Set implementation in config:
   - `"implementation": "rl_autoschedular_v3"`
2. Optionally tune transformer hyperparameters.
3. Run standard pipeline:

```bash
sbatch scripts/get_base.sh <config>
python scripts/split_json.py <config>
sbatch scripts/train.sh <config>
sbatch scripts/eval.sh <config>
```

No script changes are required beyond choosing the implementation in config.

## Recommended starting hyperparameters

- `transformer_d_model`: 256
- `transformer_nhead`: 8
- `transformer_num_layers`: 3
- `transformer_ffn_dim`: 1024
- `transformer_dropout`: 0.1
- `transformer_activation`: gelu
- `transformer_pooling`: cls
- `transformer_use_action_history_token`: false

## Measured sanity results

Lightweight end-to-end sanity was executed with one benchmark using the existing shared environment (`~/envs/mlir`) and a compact config (`bench_count=1`, `nb_iterations=5`).

- Train artifacts: `results/test_v3_sanity/v3_agent/run_0`
- Eval artifacts: `results/test_v3_sanity/v3_agent/run_1`
- Benchmark: `albert_sl128_bs1_generic_1`
- Eval final speedup: `2.1240696519147306`
- Eval cumulative reward: `0.3271687539095493`
- Eval execution time: `57371`

These runs validate end-to-end stability and artifact generation for V3 without introducing any script changes.

## Ablation smoke metrics

The following architecture-only ablation metrics were measured on CPU by timing embedding forward passes on a synthetic observation (`NumLoops=3`).

| Setting | Pooling | Layers | History token | Output size | Params | Avg embedding ms |
|---|---|---:|---|---:|---:|---:|
| A | cls | 2 | false | 690 | 7,022,271 | 3.5852 |
| B | cls | 3 | false | 690 | 8,601,791 | 4.8569 |
| C | mean | 3 | false | 690 | 8,601,791 | 6.9716 |
| D | cls | 3 | true | 256 | 8,511,679 | 4.8396 |

Interpretation:
- Going from 2 to 3 layers increases representational capacity with expected compute overhead.
- `mean` pooling is slower than `cls` in this setup.
- Enabling history token reduces external embedding size (no post-concatenation of ActionHistory).

## What is unchanged in V3

- PPO objective and update logic
- Reward function and environment step logic
- Action-space definition and masks
- Training/evaluation script interfaces

This keeps attribution clean: observed behavior changes can be tied to the encoder replacement.

## Validation checklist

- Python compile checks pass for modified files.
- Import smoke test for `rl_autoschedular_v3.model` passes.
- Implementation routing resolves `rl_autoschedular_v3` correctly.
- No baseline-package import references remain inside `rl_autoschedular_v3` Python files.

## Limitations

- Producer loop validity is inferred from non-zero static feature signals.
- Sequence length grows with `max_num_loops`, increasing memory/compute at larger settings.
- No warmup schedule was added in V3 (can be explored later if needed).

## Next experiments

1. Keep A and B as primary candidates depending on throughput vs capacity target.
2. Run longer training with B (3 layers, cls, history token off) as research default.
3. Evaluate D when final assembly aims to reduce external embedding width.
4. Evaluate on deeper loop nests to measure attention benefits.

## Ultimate-version assembly notes

To make final merging of all novelties feasible and low-risk, V3 was implemented with strict interface compatibility:

- No training/evaluation script forks were introduced.
- All transformer controls are namespaced under `transformer_*` config keys.
- Policy/value model external call contracts are unchanged.
- This allows combining V1 + V2 + V3 + V4 via composition instead of rewrites.

## References

- `docs/VERSIONS.md`
- `docs/NOVELTIES.md`
- `docs/Novelties/v1_hardware_aware_observation.md`
- `docs/Novelties/v2_shaped_reward.md`
- `docs/Novelties/v4_action_space_expansion.md`
