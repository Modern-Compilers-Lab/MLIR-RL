# V1: Hardware-Aware Observation

## Overview

Version 1 introduces **hardware-aware observation** to the RL agent's state representation. Instead of relying solely on operation and loop structure features, the agent now explicitly observes the target hardware's capabilities. This allows the RL model to make optimization decisions tailored to specific hardware characteristics.

## Problem Statement

The baseline RL agent makes optimization decisions (tiling factors, parallelization, vectorization) based only on the MLIR code structure. However, optimal transformations depend heavily on:
- **Cache hierarchy**: L1/L2/L3 cache sizes determine effective tile sizes and memory access patterns
- **CPU capabilities**: Physical/logical core counts, SIMD width, and clock speed affect vectorization potential and parallelization granularity
- **Hardware variations**: What works well on a 16-core Xeon may be suboptimal on an 8-core ARM processor

Without this information, the agent must learn optimal settings through trial and error across different hardware platforms, leading to:
- Slower convergence during training
- Potential performance regressions when deployed to different hardware
- Inability to generalize across hardware families

## Solution: Hardware-Aware Observation

V1 adds a new **Hardware Features** observation component that captures the target system's key characteristics. The agent now receives:
1. **Cache information**: L1/L2/L3 cache sizes (KB)
2. **Parallelism capacity**: Physical cores, logical cores
3. **SIMD capabilities**: SIMD width (bits), CPU clock (MHz)

These features are either:
- **Auto-detected** at runtime from `/sys/devices/` on Linux systems
- **Optionally overridden** in the JSON config file for advanced cross-hardware experiments

## Implementation Details

### Architecture Changes

#### 1. Hardware Feature Extraction (`rl_autoschedular_v1/observation.py`)

New `HardwareFeatures` class:
```python
class HardwareFeatures:
    """Extracts and normalizes hardware feature vectors."""
    
    def __init__(self, config=None):
        # Auto-detect from system if enabled
        # Otherwise use config overrides
        # Normalize all values to reasonable ranges
        
    def get_feature_vector(self):
        # Returns [l1_kb, l2_kb, l3_kb, phys_cores, log_cores, simd_width, clock_mhz]
        # All normalized to [0, 1] range
```

**Key methods:**
- `__auto_detect()`: Reads from `/proc/cpuinfo`, `/sys/devices/system/cpu/`, and lscpu
- `__get_from_config()`: Loads from JSON config fields like `hardware_l1_kb`, `hardware_physical_cores`
- `get_normalized_vector()`: Returns 7-dimensional normalized feature vector

**Detection logic:**
- If `hardware_auto_detect=true` in config, queries system files
- Uses config overrides only when an override field is explicitly set
- Detection runs once per Python process (module import time), then the same hardware vector is reused for all benchmarks in that process
- All values normalized using fixed ranges (for example L1/256, L2/4096, cores/256)

#### 2. Model Integration (`rl_autoschedular_v1/model.py`)

Updated `LSTMEmbedding` class:
```python
class LSTMEmbedding:
    def __init__(self, ...):
        self.operation_features_lstm = ...
        self.producer_features_lstm = ...
        self.hardware_features_size = HardwareFeatures.size()  # 7
        
    @property
    def output_size(self):
        # Increased to include hardware features
        return (lstm_output_size * 2) + self.hardware_features_size
```

**Integration point:**
- During forward pass, hardware features vector is concatenated after LSTM outputs
- Policy and value networks now consume `[operation_lstm_output, producer_lstm_output, hardware_features]`
- This allows the networks to condition decisions on hardware properties

#### 3. Observation Construction

Updated observation pipeline:
1. Extract operation features (unchanged)
2. Extract producer features (unchanged)
3. **NEW**: Extract hardware features from system or config
4. Concatenate all into single observation vector
5. Pass to model for embedding

### Configuration

New config fields (optional, with sensible defaults):

```json
{
  "implementation": "rl_autoschedular_v1",
  "hardware_auto_detect": true
}
```

**Behavior:**
- If `hardware_auto_detect=true`, system is queried for all fields
- Hardware detection is done once per process, not once per dataset file
- Measured execution time and speedup still come from real code execution on the host CPU
- Manual overrides (`hardware_l1_kb`, `hardware_l2_kb`, `hardware_l3_kb`, `hardware_physical_cores`, `hardware_logical_cores`, `hardware_simd_width`, `hardware_clock_mhz`) remain available but are usually unnecessary on a single fixed HPC machine

### Files Modified

| File | Changes |
|------|---------|
| `rl_autoschedular_v1/observation.py` | New `HardwareFeatures` class for extraction/normalization |
| `rl_autoschedular_v1/model.py` | Updated `LSTMEmbedding.output_size`, hardware vector concatenation |
| `utils/config.py` | Added 8 hardware-related config fields with defaults |
| `config/example.json` | Documented hardware config options |
| `utils/implementation.py` | Added v1 implementation routing |
| `scripts/*.sh` | Config-aware implementation resolution |

## How to Use

### Option 1: Auto-Detect Hardware (Recommended for Single HPC)

```json
{
  "implementation": "rl_autoschedular_v1",
  "hardware_auto_detect": true
}
```

The system will automatically detect all hardware features at runtime.

### Option 2: Explicit Overrides (Advanced)

```json
{
  "implementation": "rl_autoschedular_v1",
  "hardware_auto_detect": false,
  "hardware_l1_kb": 32,
  "hardware_l2_kb": 256,
  "hardware_l3_kb": 8192,
  "hardware_physical_cores": 16,
  "hardware_logical_cores": 32,
  "hardware_simd_width": 256,
  "hardware_clock_mhz": 2400
}
```

Use overrides only when you intentionally need to simulate/condition on a different hardware profile.
For single-machine train/eval, prefer auto-detect without overrides.

## Execution-Time Impact

- Hardware detection does not directly change the execution timer.
- Execution time is measured later when transformed code is executed on the host CPU.
- Hardware features affect policy decisions indirectly (which transformations are chosen), which can then change measured speedup.

### Running the Pipeline

Once config is set:

```bash
# Prepare base execution times
sbatch scripts/get_base.sh config/train1.json

# Split benchmark data
python scripts/split_json.py config/train1.json

# Train with hardware-aware observation
sbatch scripts/train.sh config/train1.json

# Evaluate
sbatch scripts/eval.sh config/train1.json

# Compare versions in dashboard
streamlit run dashboard/dashboard.py --server.fileWatcherType none
```

All scripts will automatically use V1 implementation based on the config.

## Expected Benefits

1. **Faster Convergence**: Agent learns hardware-specific optimal tile sizes and parallelization strategies earlier
2. **Better Generalization**: Model can adapt to different hardware without complete retraining
3. **Reduced Trial-and-Error**: Explicit hardware knowledge reduces exploration of suboptimal regions
4. **Interpretability**: Can analyze how agent weights hardware features when making decisions

## Validation Results

✅ Python compile checks passed
✅ Import test successful (`rl_autoschedular_v1.model`)
✅ Implementation routing verified
✅ No baseline package references inside V1 code
✅ Hardware feature extraction working on test systems

## Limitations and Future Work

### Current Limitations
- Hardware features are 7-dimensional (may miss exotic features like specialized accelerators)
- Auto-detection uses Linux-specific paths (`/sys/devices/`)
- Assumes homogeneous multi-core (all cores same frequency)
- Does not model memory bandwidth explicitly

### Future Improvements
- Add memory bandwidth probing
- Support ARM, macOS, Windows hardware detection
- Include accelerator info (GPU, TPU, etc.) if available
- Dynamic hardware reconfiguration during training
- Learn feature importance via attention mechanisms

## Troubleshooting

### Hardware Features Not Detected

If auto-detection fails, check system files:
```bash
cat /proc/cpuinfo | grep "cache size"
lscpu
```

Then explicitly set config fields if needed.

### Wrong Hardware Features Detected

The auto-detection may be incorrect for heterogeneous systems. Verify with:
```bash
python -c "from rl_autoschedular_v1.observation import HARDWARE_VECTOR; print(HARDWARE_VECTOR)"
```

If incorrect, use explicit config values.

### Model Not Using Hardware Features

Verify hardware features are in observation:
```bash
python -c "from rl_autoschedular_v1.observation import Observation; o = Observation(...); print('Hardware in obs:', o.hardware_features is not None)"
```

Check model embedding includes hardware size:
```bash
python -c "from rl_autoschedular_v1.model import PPOModel; m = PPOModel(...); print('Embedding size:', m.embedding_output_size)"
```

## References

- [VERSIONS.md](../VERSIONS.md) - Version history and validation details
- [RL_AGENT_TUTORIAL.md](../RL_AGENT_TUTORIAL.md) - General RL agent architecture
- [TRAINING_GUIDE.md](../TRAINING_GUIDE.md) - Training workflow
