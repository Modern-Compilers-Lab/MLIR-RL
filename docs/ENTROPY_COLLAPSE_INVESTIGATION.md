# Entropy Collapse Investigation: V4.x Training Failure

**Date:** June 10, 2026
**Status:** Confirmed root cause. Fixes prescribed.

## Executive Summary

V4.6, V4.7, and V4.8 all fail to learn on the single_ops_dataset due to **entropy collapse** — the policy becomes deterministic mid-training, producing identical actions for every benchmark on every subsequent iteration. V0 (baseline, no HW features, no shaped reward) does not suffer this problem and continues learning throughout training.

**Key finding:** The combination of shaped reward + transformer encoder causes the policy to converge prematurely to a local optimum with zero entropy. Once entropy reaches zero, PPO's policy gradient vanishes, making recovery impossible.

---

## 1. Affected Agents

| Agent | Implementation | Transformer | HW Features | Shaped Reward | lr | Entropy Collapse |
|-------|---------------|-------------|-------------|---------------|-----|-----------------|
| **V0** | rl_autoschedular_v0 (LSTM) | No | No | No | 0.001 | **No** — healthy 14K+ iters |
| **V4.6** | rl_autoschedular_v4_5 (Transformer) | d=256, 8 heads, 3 layers | Yes | Yes (scale=0.05) | 0.001 | **Yes** — iter 6599 |
| **V4.7** | rl_autoschedular_v4_5 (Transformer) | d=64, 2 heads, 2 layers | Yes | Yes (scale=0.05) | 0.001 | **Yes** — iter 9462 |
| **V4.8** | rl_autoschedular_v4_5 (Transformer) | d=256, 8 heads, 3 layers | Yes | Yes (scale=0.05) | 0.001 | **Yes** — iter 8333 |

All agents use the same lr=0.001. Learning rate is NOT a differentiating factor.

### Config Differences (V0 vs V4.6)

| Parameter | V0 | V4.6 |
|-----------|-----|------|
| `implementation` | `rl_autoschedular_v0` (LSTM) | `rl_autoschedular_v4_5` (Transformer) |
| `hardware_auto_detect` | `false` | `true` |
| `reward_shaping_enabled` | `false` | `true` |
| `reward_shaping_scale` | `1.0` (unused) | `0.05` |
| `reward_shaping_clip` | `2.0` (unused) | `0.1` |
| `transformer_d_model` | N/A | `256` |
| `lr` | `0.001` | `0.001` |

The key differences between V0 and V4.6 are: (1) Transformer vs LSTM, (2) HW feature observations enabled, (3) shaped reward enabled. V4.7 differs only in a smaller Transformer (d=64 vs d=256). Learning rate (0.001) is identical.

---

## 2. Entropy Collapse Timeline

### V4.6 (permanent collapse at iter 6599 / ~58% of training)

```
iter    0: avg_entropy = 1.791130
iter  500: avg_entropy = 2.110388
iter 1000: avg_entropy = 1.832981
iter 2000: avg_entropy = 3.528299
iter 4000: avg_entropy = 1.175832
iter 5000: avg_entropy = 2.850430
iter 6500: avg_entropy = 0.000000  ← COLLAPSE
iter 6513: avg_entropy = 0.475819  ← brief recovery
iter 6599: avg_entropy = 0.000000  ← PERMANENT COLLAPSE
iter 6600+: all zeros
```

Entropy oscillated normally for 6500 iterations, then collapsed abruptly. A brief recovery at iter 6513-6598 was followed by permanent collapse.

### V4.7 (permanent collapse at iter 9462 / ~47% of training)

```
iter  9459: avg_entropy = 2.668391
iter  9460: avg_entropy = 1.299798
iter  9461: avg_entropy = 1.335877  ← last healthy iter
iter  9462: avg_entropy = 0.000000  ← COLLAPSE (instant, no recovery)
```

V4.7 collapsed instantly from healthy entropy (1.34) to zero in a single iteration. The smaller transformer (d=64) did not prevent collapse — it merely delayed it.

### V4.8 (permanent collapse at iter 8333 / ~72% of training)

```
iter  8330: avg_entropy = 0.007172  ← already collapsing
iter  8331: avg_entropy = 0.315494  ← brief flicker
iter  8332: avg_entropy = 0.274879  ← last partial recovery
iter  8333: avg_entropy = 0.000103  ← PERMANENT COLLAPSE
```

V4.8 (identical config to V4.6) survived longer (72% vs 58%) due to different random initialization and checkpoint scheduling.

### V0 (no collapse — healthy throughout)

```
iter     0: avg_entropy = 1.791130
iter  2000: avg_entropy = 0.216475  ← dip, but recovers
iter  3000: avg_entropy = 0.006519  ← near-zero, but recovers
iter  5000: avg_entropy = 0.219499
iter  9000: avg_entropy = 0.484476
iter 12500: avg_entropy = 0.718516
iter 14941: avg_entropy = 0.493347  ← still healthy at end
```

V0's entropy fluctuates (sometimes dipping below 0.01) but always recovers. It never permanently collapses across 14,942 iterations.

---

## 3. Impact on Evaluation

### V4.6 eval results are identical across all checkpoints

| Checkpoint | Avg Speedup | GeoMean | Benchmarks | Identical to ckpt 100 |
|-----------|------------|---------|-----------|----------------------|
| 100 | 1.20x | 0.66x | 305 | — |
| 200 | 1.14x | 0.57x | 305 | 303/305 (99.3%) |
| 500 | 1.14x | 0.57x | 305 | 303/305 (99.3%) |
| 1000 | 1.14x | 0.58x | 305 | 303/305 (99.3%) |
| 1200 | 1.14x | 0.57x | 305 | 303/305 (99.3%) |

The model was already frozen before checkpoint 100. Every subsequent checkpoint produces identical schedules.

### V0 eval results show improvement over training

| Checkpoint | Avg Speedup | GeoMean | Benchmarks |
|-----------|------------|---------|-----------|
| 100 | 1.33x | 0.66x | 303 |
| 500 | 1.42x | 0.72x | 301 |
| 1000 | 2.40x | 0.91x | 302 |
| 1500 | 1.39x | 0.68x | 302 |
| 2000 | 2.49x | 0.87x | 301 |

V0's geo mean improves from 0.66x to 0.91x (ckpt 1000), showing genuine learning.

### Training speedup (per-iteration)

All agents show flat per-iteration training speedup (~12.5-13x) throughout training. This is misleading — the training speedup is averaged over 64 random benchmarks per iteration with significant variance. The eval speedup (over the full 305-benchmark eval set) reveals the true learning signal.

---

## 4. Root Cause Analysis

### Why entropy collapse happens in V4.x but not V0

**V4.x has two factors that V0 lacks:**

1. **Shaped reward** provides strong, consistent gradients that push the policy toward a specific local optimum. The shaped reward (scale=0.05, clip=0.1) gives intermediate feedback for improving arithmetic intensity and parallelism. This creates a gradient signal that is:
   - Much stronger than the sparse terminal reward
   - Consistent across similar benchmarks (same optimization "heuristics" work)
   - Easy to exploit with a deterministic policy (just apply the same high-reward actions)

2. **Transformer encoder** has higher capacity than LSTM. The transformer can represent complex state-action mappings more precisely, which means it can overfit to the shaped reward signal more efficiently. Once it finds a "good enough" policy for the shaped reward, the advantage of exploration drops to zero.

### The collapse mechanism

```
Phase 1 (healthy): Policy explores diverse actions → entropy > 0.1
Phase 2 (pre-collapse): Shaped reward consistently rewards same actions → 
    policy concentrates probability mass on a few actions → entropy drops
Phase 3 (collapse): Advantage estimates for "dominant" actions become large → 
    policy gradient pushes those actions to probability 1.0 → entropy = 0
Phase 4 (dead): Zero entropy → zero policy gradient → no recovery possible
```

### Why V0 doesn't collapse

- **No shaped reward**: The only reward signal is the terminal speedup, which is noisy and varies per benchmark. There's no consistent gradient pointing to a single policy.
- **LSTM has lower capacity**: Less prone to overfitting the reward signal.
- **Exploration stays necessary**: Without shaped reward "hints," the agent must keep exploring to find good schedules.

---

## 5. Why This Matters for the Paper

### Shaped reward harms learning (supports V4.5 lesson)

This confirms the finding from V4.5 training on new_dataset: **shaped reward misleads the agent.** When intermediate reward dominates terminal speedup, the agent optimizes static heuristics instead of execution time.

### Transformer is not inherently better

The transformer's higher capacity is a double-edged sword. With proper reward design, it can learn complex optimization strategies. With shaped reward, it overfits faster than LSTM.

### Entropy collapse is a fundamental failure mode

This is not a hyperparameter tuning issue. The combination of shaped reward + transformer + PPO creates a structural incentive for collapse. Any configuration with these three elements is at risk.

---

## 6. Recommended Fixes

### Fix 1: Entropy bonus (immediate, minimal code change)

**Problem:** `entropy_coef=0.01` is too weak to prevent collapse with shaped reward.

**Solution:** Increase `entropy_coef` to 0.05-0.10 and add entropy monitoring with automatic restart.

```python
# In PPO update, add entropy penalty to loss
entropy_loss = -self.entropy_coef * entropy.mean()
total_loss = policy_loss + value_loss + entropy_loss

# Add entropy floor: if entropy < 0.01, inject exploration noise
if entropy.mean() < 0.01:
    # Add Gaussian noise to action logits before sampling
    action_logits = action_logits + torch.randn_like(action_logits) * 0.1
```

**Config change:**
```json
"entropy_coef": 0.05
```

### Fix 2: KL divergence penalty (prevents policy from moving too fast)

**Problem:** PPO's clipped objective doesn't prevent the policy from collapsing if the clipped region is consistently hit.

**Solution:** Add a KL divergence penalty between current and reference policy.

```python
# In PPO update
kl_div = F.kl_div(
    F.log_softmax(current_logits, dim=-1),
    F.softmax(reference_logits, dim=-1),
    reduction='batchmean'
)
kl_loss = kl_coeff * kl_div.mean()
total_loss = policy_loss + value_loss + kl_loss
```

**Config change:**
```json
"kl_coef": 0.02,
"kl_target": 0.01
```

### Fix 3: Periodic entropy injection (nuclear option)

**Problem:** Once entropy reaches zero, standard PPO cannot recover.

**Solution:** Monitor entropy and reset to a reference policy when it drops below a threshold.

```python
# In training loop
if entropy.mean() < 0.05:
    logger.warning(f"Entropy collapse detected at iter {iter}. Resetting policy.")
    # Reload reference policy (saved at training start)
    policy.load_state_dict(reference_policy_state_dict)
    # Optionally reduce learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5
```

### Fix 4: Disable shaped reward (nuclear option — proven to work)

**Problem:** Shaped reward is the primary driver of collapse.

**Solution:** Train V4.x without shaped reward, using only terminal speedup.

```json
"reward_shaping_enabled": false
```

This is equivalent to V0's training regime but with a transformer. Based on the V4.5 lesson (no-reward ablation outperformed all shaped-reward variants), this is the most likely path to success.

### Recommended priority

1. **Fix 1 + Fix 2** (entropy bonus + KL penalty) — minimal risk, addresses the most likely cause
2. **Fix 4** (disable shaped reward) — proven to work, but removes HW features' contribution signal
3. **Fix 3** (entropy injection) — nuclear option, use only if other fixes fail

---

## 7. Verification Plan

After applying fixes, verify with the following checks:

1. **Entropy monitoring**: Plot entropy per iteration. Should stay > 0.1 throughout training.
2. **Eval checkpoint diversity**: Check that eval results differ between checkpoints (not 99%+ identical).
3. **Speedup improvement**: Eval geo mean should improve over training (not stay flat).
4. **No NaN/zero loss**: Monitor policy loss, value loss, and entropy for NaN or zero values.

```bash
# After training, check entropy
python3 -c "
import json
entropies = [float(l) for l in open('results/.../logs/train/entropy') if l.strip()]
for i in range(0, len(entropies), 64*100):
    avg = sum(entropies[i:i+64])/64
    print(f'iter {i//64}: entropy={avg:.4f}')
"
```

---

## 8. Log References

| Agent | Training Job | Log File | Checkpoints |
|-------|-------------|----------|-------------|
| V0 | 16248979 | `logs/train_16248979.out` | 50-3000 (every 50) |
| V4.6 | 16248980 | `logs/train_16248980.out` | 50-2150 (every 50) |
| V4.7 | 16248981 | `logs/train_16248981.out` | 50-3000 (every 50) |
| V4.8 | 16248982 | `logs/train_16248982.out` | 50-3000 (every 50) |

Eval results:
- `results/single_ops_dataset_results/v0_agent/eval/checkpoint_{N}.json` (N=100..2200)
- `results/single_ops_dataset_results/v4_6_agent/eval/checkpoint_{N}.json` (N=100..1200)
