# V2: Shaped Reward

## Overview

Version 2 introduces **reward shaping** to guide the RL agent toward meaningful optimizations earlier in training. Rather than only rewarding final execution speedup at episode end, V2 provides dense intermediate rewards based on the *quality* of each transformation applied, enabling faster policy learning and more stable convergence.

## Problem Statement

The baseline RL agent only receives reward at the end of an episode:
- **Terminal reward**: Log-ratio of original vs. final execution time (`log₁₀(original/final)`)
- **Sparse feedback**: Agent doesn't know if a transformation was locally good until the full episode completes
- **Slow convergence**: PPO must discover effective transformation sequences through random exploration
- **Unstable training**: Variance in final rewards makes gradient estimates noisy

This sparse feedback structure means:
- Early episodes provide almost no signal (random transforms rarely improve performance)
- The agent wastes computation exploring bad transformation sequences to completion
- Training curves are noisy and require many iterations to stabilize
- Small benchmark variations cause large reward swings

## Solution: Shaped Reward

V2 adds **dense intermediate rewards** that reinforce locally good decisions:

1. **Static Efficiency Score**: Each operation's potential for optimization based on:
   - Arithmetic intensity (operations per byte loaded)
   - Vectorizability (is the operation vectorizable?)
   - Parallel loop potential (degree of parallelizable nesting)

2. **Per-Step Shaped Reward**: Quantifies if the current transformation improved the operation's efficiency score

3. **Blended Final Reward**: Combines shaped signals with terminal execution reward to guide learning

The result: Agent receives positive signals **immediately after good transformations**, speeding convergence and reducing noise.

## Implementation Details

### Reward Calculation Pipeline

#### 1. Static Efficiency Score (`rl_autoschedular_v2/env.py`)

New method `__static_efficiency_score()`:
```python
def __static_efficiency_score(operation):
    """
    Computes operation's static optimization potential from features.
    Returns score ∈ [0, 1] combining:
    - Arithmetic intensity ratio
    - Vectorizability (binary: 0 or 1)
    - Parallel loop ratio
    """
    # Estimate arithmetic intensity (ops per byte moved)
    ai = __estimate_arithmetic_intensity(operation)
    
    # Check vectorizability from operation type
    vectorizable = operation.is_floating_point and \
                   operation.num_operands >= 2
    
    # Estimate parallel loop ratio from nesting depth
    parallel_ratio = min(parallel_loops / total_loops, 1.0)
    
    # Weighted combination
    score = (weight_ai * ai_norm +
             weight_vec * float(vectorizable) +
             weight_par * parallel_ratio)
    
    return min(score, 1.0)
```

**Components:**

- **Arithmetic Intensity Estimation**: 
  - Approximates FLOPs and memory bytes accessed from operation metadata
  - Higher AI → more compute-bound → lower priority for vectorization/tiling changes
  - Formula: `FLOPs / Bytes_Accessed`, normalized to [0, 1]

- **Vectorizability Check**:
  - True for floating-point operations with ≥2 operands (can be vectorized)
  - False for scalar or complex operations
  - Binary signal (0 or 1)

- **Parallel Loop Ratio**:
  - Fraction of loops that are parallelizable
  - Higher ratio → more potential for multi-threaded speedup
  - Range: [0, 1]

#### 2. Shaped Reward per Transformation (`__shaped_reward()`)

New method that computes the reward for applying a transformation:
```python
def __shaped_reward(old_operation, new_operation, action):
    """
    Computes intermediate reward for this transformation.
    Positive if efficiency improves, negative otherwise.
    """
    old_score = self.__static_efficiency_score(old_operation)
    new_score = self.__static_efficiency_score(new_operation)
    
    delta = new_score - old_score
    
    # Clip to stable range and scale
    clipped_delta = np.clip(delta, 
                            -reward_shaping_clip, 
                            reward_shaping_clip)
    
    shaped_reward = reward_shaping_scale * clipped_delta
    
    # Bonus for vectorization actions
    if action_type == VECTORIZE and new_score > old_score:
        shaped_reward += reward_shaping_vectorization_bonus
    
    return shaped_reward
```

**Key features:**
- **Delta-based**: Rewards improvement relative to previous state
- **Bounded**: Clipped to `[-clip, +clip]` to prevent reward spikes
- **Scaled**: Multiplied by configurable scale factor
- **Vectorization bonus**: Extra reward for successful vectorization (often impactful)

#### 3. Per-Step Reward Recording

Modified `step()` method to store shaped reward:
```python
def step(self, action):
    # Apply transformation
    new_operation = self.__apply_transformation(action)
    
    # Compute shaped reward immediately
    shaped_reward = self.__shaped_reward(
        self.current_operation,
        new_operation,
        action
    )
    
    # Store in action extras for later aggregation
    action.extras['shaped_reward'] = shaped_reward
    
    self.current_operation = new_operation
    return shaped_reward, ...
```

#### 4. Final Reward Aggregation (`__apply_sequence()`)

Modified to blend shaped and terminal rewards:
```python
# Terminal reward (unchanged): execution time speedup
terminal_reward = math.log10(old_time / new_time)

# Sum shaped rewards from all steps
total_shaped_reward = sum(
    action.extras.get('shaped_reward', 0)
    for action in trajectory
)

# Blend with weights
final_reward = total_shaped_reward + terminal_reward

# Apply penalty for out-of-bounds access, etc.
if failed:
    final_reward = FAILURE_PENALTY
```

### Configuration

New config fields control reward shaping (all optional with sensible defaults):

```json
{
  "implementation": "rl_autoschedular_v2",
  "reward_shaping_enabled": true,
  "reward_shaping_scale": 1.0,
  "reward_shaping_clip": 2.0,
  "reward_shaping_weight_ai": 1.0,
  "reward_shaping_weight_vectorizable": 0.1,
  "reward_shaping_weight_parallel": 0.1,
  "reward_shaping_vectorization_bonus": 0.2
}
```

| Field | Default | Purpose |
|-------|---------|---------|
| `reward_shaping_enabled` | `true` | Enable/disable shaped rewards entirely |
| `reward_shaping_scale` | `1.0` | Multiplier on per-step rewards (0.5-2.0 typical) |
| `reward_shaping_clip` | `2.0` | Max magnitude of per-step reward (prevents outliers) |
| `reward_shaping_weight_ai` | `1.0` | Weight of arithmetic intensity in efficiency score |
| `reward_shaping_weight_vectorizable` | `0.1` | Weight of vectorizability flag |
| `reward_shaping_weight_parallel` | `0.1` | Weight of parallel loop ratio |
| `reward_shaping_vectorization_bonus` | `0.2` | Extra reward for VECTORIZE actions that improve score |

### Files Modified

| File | Changes |
|------|---------|
| `rl_autoschedular_v2/env.py` | Added `__shaped_reward()`, `__static_efficiency_score()`, `__estimate_arithmetic_intensity()`, `__estimate_bytes_moved()` |
| `rl_autoschedular_v2/env.py` | Modified `step()` to record shaped rewards in action.extras |
| `rl_autoschedular_v2/env.py` | Modified `__apply_sequence()` to blend shaped + terminal rewards |
| `utils/config.py` | Added 7 reward shaping config fields with defaults |
| `config/example.json` | Documented shaped reward fields |
| `utils/implementation.py` | Added v2 implementation routing |
| `scripts/*.sh` | Config-aware implementation resolution |

## How to Use

### Option 1: Default Shaped Reward (Recommended)

```json
{
  "implementation": "rl_autoschedular_v2",
  "reward_shaping_enabled": true
}
```

Uses default shaped reward with recommended weights and scaling. Simple and effective for most benchmarks.

### Option 2: Tuned for Specific Benchmark Family

```json
{
  "implementation": "rl_autoschedular_v2",
  "reward_shaping_enabled": true,
  "reward_shaping_scale": 1.5,
  "reward_shaping_weight_ai": 0.5,
  "reward_shaping_weight_vectorizable": 0.3,
  "reward_shaping_vectorization_bonus": 0.5
}
```

Adjust scales and weights based on benchmark characteristics:
- **Compute-heavy** (high AI): Reduce `weight_ai`, increase `weight_vectorizable`
- **Memory-bound** (low AI): Increase `weight_ai`, reduce `weight_vectorizable`
- **Parallelizable loops**: Increase `weight_parallel` and `weight_vectorizable`

### Option 3: Disable Shaped Reward (Baseline Comparison)

```json
{
  "implementation": "rl_autoschedular_v2",
  "reward_shaping_enabled": false
}
```

Uses only terminal rewards (equivalent to baseline). Useful for A/B testing.

### Running the Pipeline

```bash
# Prepare base execution times
sbatch scripts/get_base.sh config/train1.json

# Split benchmark data
python scripts/split_json.py config/train1.json

# Train with shaped rewards
sbatch scripts/train.sh config/train1.json

# Evaluate and compare
sbatch scripts/eval.sh config/train1.json

# View results in dashboard
streamlit run dashboard/dashboard.py --server.fileWatcherType none
```

All scripts will automatically use V2 with shaped reward based on the config.

## Expected Benefits

1. **Faster Convergence**: Dense rewards guide learning toward good transformations faster
2. **Lower Variance**: Intermediate rewards smooth out random noise in final episode outcomes
3. **Interpretable Learning**: Shaped rewards align with human intuition (vectorize compute-intensive ops, parallelize loops, tile for cache)
4. **Reduced Training Time**: Fewer random exploration episodes needed before meaningful progress
5. **Better Sample Efficiency**: Each training step provides more informative gradient signals

## Validation Results

✅ Python compile checks passed for V2 and routing files
✅ Implementation routing verified (`rl_autoschedular_v2 → v2_agent / v2`)
✅ Import test successful (`rl_autoschedular_v2.env`)
✅ Config defaults loaded correctly
✅ No baseline package references inside V2 code

## Expected vs. Actual

### Hypothesis
- Shaped reward should provide denser learning signals, leading to faster convergence
- Vectorization bonus should particularly improve vectorization action selection early in training

### How to Validate
1. Train baseline, V1, and V2 on same benchmark suite
2. Compare learning curves:
   - X-axis: Training step or episode
   - Y-axis: Moving average of episode return
3. Compare final speedups achieved within same training budget
4. Analyze which actions the agent selects more frequently under shaped reward

### Metrics to Track
- **Convergence speed**: Iterations to reach N% of asymptotic reward
- **Final performance**: Speedup achieved after fixed training steps
- **Stability**: Standard deviation of rolling average return
- **Sample efficiency**: Reward per gradient step

## Limitations and Future Work

### Current Limitations
- Static efficiency scores ignore actual data access patterns
- Arithmetic intensity estimated from operation type, not actual kernel structure
- No model-specific profiling (may be cheap for some ops, expensive for others)
- Weights are fixed; no adaptive tuning during training

### Future Improvements
- **Profiling-based scoring**: Measure actual arithmetic intensity and bandwidth during initial benchmark runs
- **Adaptive weighting**: Use attention mechanisms to learn importance of each component
- **Operation-specific bonuses**: Different bonus values for different operation types
- **Execution prediction**: Predict post-transformation execution time and reward directly
- **Curriculum learning**: Start with high shaping weight, gradually shift to terminal reward

## Troubleshooting

### Reward Shaping Not Improving Training

1. **Check if enabled**: Verify `reward_shaping_enabled=true` in your config and logs
2. **Adjust scale**: Try increasing `reward_shaping_scale` (1.5-2.0) to make shaped rewards more prominent
3. **Check weights**: Ensure weights are positive and balanced
4. **Visualize**: Log shaped vs terminal rewards per episode to understand ratio

### Training Diverges with Shaped Reward

1. **Reduce scale**: Try `reward_shaping_scale=0.5` to dampen shaped signals
2. **Increase clip**: Try `reward_shaping_clip=5.0` to allow larger rewards
3. **Check weights**: Ensure no single weight dominates (sum to ~1.0)
4. **Baseline comparison**: Run with `reward_shaping_enabled=false` to isolate issue

### Vectorization Not Being Selected

1. **Increase vectorization bonus**: Try `reward_shaping_vectorization_bonus=1.0`
2. **Lower weight_vectorizable**: Increase impact of pure vectorization action
3. **Check operation types**: Ensure benchmarks have vectorizable operations (FP floating-point ops)

## References

- [VERSIONS.md](../VERSIONS.md) - Version history and validation details
- [RL_AGENT_TUTORIAL.md](../RL_AGENT_TUTORIAL.md) - General RL agent architecture and reward structure
- [TRAINING_GUIDE.md](../TRAINING_GUIDE.md) - Training workflow and hyperparameter tuning
- PPO Algorithm: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
