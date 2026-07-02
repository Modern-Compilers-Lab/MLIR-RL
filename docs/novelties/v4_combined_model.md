# V4: Combined Enhancements (V1 + V2 + V3)

## Overview

Version 4 represents the integration of all early-stage enhancements into a single, comprehensive RL agent. It combines the **Hardware-Aware Observation** (V1), **Shaped Reward** (V2), and the **Transformer Loop-Nest Encoder** (V3) to maximize scheduling performance, cross-hardware generalization, and training stability.

## Integrated Components

V4 brings together the following novelties:

1. **Hardware-Aware Observation (from V1)**: The agent receives explicit features about the target hardware (L1/L2/L3 cache sizes, physical/logical core counts, SIMD width, clock speed). This allows the schedule to adapt its tiling and parallelization optimally to different microarchitectures.
2. **Shaped Reward (from V2)**: Instead of relying solely on sparse, delayed execution time improvements, V4 uses intermediate reward shaping (based on heuristics like arithmetic intensity and vectorizability). This guides the agent during early training steps and accelerates convergence.
3. **Transformer Loop-Nest Encoder (from V3)**: The underlying MLIR loop structures are processed using an attention-based sequence encoder. This enables the agent to better capture nested dependencies and complex data-flow patterns compared to simple flattened MLP layers.

## Rationale for Combination

V1, V2, and V3 each target a distinct orthogonal component of the RL pipeline—State Observation, Reward Signal, and Neural Architecture, respectively. Combining them yields powerful synergistic effects:

- **Architecture + Hardware (V3 + V1)**: The Transformer encoder creates complex representations of loop nests. Supplementing this robust embedding with strict hardware boundary characteristics ensures that memory-hierarchy limits constrain the generated representations properly.
- **Architecture + Training Stability (V3 + V2)**: Transformers can be notoriously difficult and sample-inefficient to train purely with Delayed RL Sparse Rewards. The dense, intermediate shaped rewards provide the step-by-step gradients necessary to properly train the deep encoder out of random initialization.

## Configuration and Setup

The V4 agent resides in the standalone `rl_autoschedular_v4` package.

To use V4, ensure your JSON config contains:

```json
{
  "implementation": "rl_autoschedular_v4",
  "hardware_auto_detect": true,
  "reward_shaping_enabled": true,
  "reward_shaping_scale": 0.5
}
```
