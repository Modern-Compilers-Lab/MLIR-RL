# V4: Action Space Expansion (Pad + Pack + Unroll)

## Overview

Version 4 expands the RL agent's action space from 6 to 9 actions by adding three new MLIR transform dialect operations: **Pad**, **Pack**, and **Unroll**. These fine-grained transformations complement the existing high-level loop transformations and give the agent finer control over memory layout, padding, and instruction-level parallelism.

## Problem Statement

The baseline action space is limited to coarse-grained loop transformations:
- **Tiling (T)**, **TiledParallelization (TP)**, **TiledFusion (TPF)**, **Interchange (I)**, **Vectorization (V)**, and **NoTransformation (NT)**

While powerful, these actions miss opportunities for:
- **Memory alignment and padding**: Operations on non-power-of-two dimensions can suffer from misaligned memory accesses and inefficient vector loads
- **Data layout optimization**: Reorganizing data into tiled/packed formats can dramatically improve cache locality for certain access patterns
- **Instruction-level parallelism**: Loop unrolling exposes more independent instructions to the backend scheduler, reducing loop overhead and enabling better register allocation

Without these transformations, the agent cannot express schedules that combine high-level tiling with low-level unrolling or memory layout changes, leaving performance on the table.

## Solution: Expanded Action Space

V4 adds three new actions:

### 1. Pad (`P`)

Pads specific dimensions of a structured operation to multiples of powers of 2. This ensures:
- **Aligned memory accesses** for vectorization
- **Power-of-two tile sizes** that work well with cache lines
- **Reduced boundary-condition overhead** when combined with tiling

**Parameterization:** Per-dimension categorical parameter:
- `0` = do not pad this dimension
- `1` = pad to multiple of 2
- `2` = pad to multiple of 4
- `3` = pad to multiple of 8

### 2. Pack (`PK`)

Packs a structured operation into a blocked/tiled data layout using `transform.structured.pack`. This is particularly effective for:
- **Matrix multiplication** and convolution-like access patterns
- **Improving cache locality** by grouping accesses into contiguous blocks
- **Enabling tiled execution** on the packed layout

**Parameterization:** Per-dimension categorical parameter (same encoding as Tiling):
- `0` = do not pack this dimension
- `1` = pack size 1
- `2` = pack size 2
- `3` = pack size 4
- etc.

### 3. Unroll (`U`)

Unrolls loops to expose instruction-level parallelism. Unlike the other actions, Unroll is implemented by:
1. First tiling the operation with `tile_sizes = bound // factor`
2. Then unrolling each resulting loop with `transform.loop.unroll`

This approach is **terminal** because unrolling duplicates the tagged operation, making subsequent tag-based transforms ambiguous (multiple ops match the same tag).

**Parameterization:** Single categorical parameter:
- `0` = unroll factor 2
- `1` = unroll factor 4
- `2` = unroll factor 8

**Restriction:** Only allowed when `producer_tag is None` (no pending producer fusion), since losing the tag would break fusion.

## Implementation Details

### Architecture Changes

#### 1. Transform Dialect Builders (`rl_autoschedular_v4/transforms.py`)

**`transform_pad()`:**
```python
def transform_pad(code, operation_tag, padding_dimensions, pad_to_multiple_of):
    # Matches tagged op, applies structured.pad with padding_dimensions
    # and pad_to_multiple_of, then re-annotates the tag on the padded op
```

Key details:
- Returns 3 results from `structured.pad`: `padded_op`, `pad_op`, `copy_op`
- Uses `pad_to_multiple_of [a, b, ...] {padding_dimensions = [i, j, ...]}` syntax (MLIR 19.x transform dialect)
- Re-annotates tag because pad creates a new op

**`transform_pack()`:**
```python
def transform_pack(code, operation_tag, packed_sizes):
    # Matches tagged op, applies structured.pack with packed_sizes,
    # then re-annotates the tag on the packed op
```

Key details:
- Uses `structured.pack %op packed_sizes = [a, b, ...]`
- Re-annotates tag because pack creates a new op with different iterator structure

**`transform_unroll()`:**
```python
def transform_unroll(code, operation_tag, tile_sizes, factor):
    # 1. Tiles using structured.tile_using_for with tile_sizes
    # 2. Unrolls each resulting loop handle with loop.unroll
    # Unrolls innermost loops first to avoid handle invalidation
```

Key details:
- Tile sizes are computed as `loop.upper_bound // factor` per dimension
- Loops are unrolled in **reverse order** (innermost first) because unrolling an outer loop invalidates handles to inner loops
- No tag re-annotation needed because `tile_using_for` preserves the tag on the tiled op

#### 2. Pad Action (`rl_autoschedular_v4/actions/pad.py`)

```python
class Pad(Action):
    symbol = 'P'
    
    def __init__(self, parameters, state=None, **extras):
        # Convert param index to multiple: 1->2, 2->4, 3->8
        # Only dimensions with non-zero parameters are padded
```

**Key features:**
- `params_size() = max_num_loops`
- `network_output_size() = max_num_loops * (num_pad_multiples + 1)`
- `action_mask()` ensures only valid multiples are proposed (multiples must be <= loop bound)
- `update_features()` rounds loop upper bounds up to the chosen multiple

#### 3. Pack Action (`rl_autoschedular_v4/actions/pack.py`)

```python
class Pack(Action):
    symbol = 'PK'
```

**Key features:**
- Reuses the same parameter encoding as Tiling (powers of 2)
- `params_size() = max_num_loops`
- `network_output_size() = max_num_loops * (num_tile_sizes + 1)`
- `action_mask()` ensures pack sizes divide the loop bound evenly
- `update_features()` updates loop bounds to `ceil(orig / pack_size)`

#### 4. Unroll Action (`rl_autoschedular_v4/actions/unroll.py`)

```python
class Unroll(Action):
    symbol = 'U'
    terminal = True  # Cannot apply further tag-based transforms after unroll
```

**Key features:**
- `params_size() = 1`
- `network_output_size() = num_unroll_factors`
- `is_allowed()` returns `True` only if `state.producer_tag is None`
- `action_mask()` checks that all loop bounds are divisible by the factor
- Computes `tile_sizes = [bound // factor for bound in loop_bounds]` and stores in `extras['tile_sizes']`
- `update_features()` updates loop bounds to the tile size (bound // factor)

#### 5. Action Space Registration (`rl_autoschedular_v4/actions/__init__.py`)

```python
class ActionSpace:
    supported_actions = [
        NoTransformation,   # NT
        Tiling,             # T
        TiledParallelization,  # TP
        TiledFusion,        # TPF
        Interchange,        # I
        Vectorization,      # V
        Pad,                # P   (NEW)
        Pack,               # PK  (NEW)
        Unroll,             # U   (NEW)
    ]
```

Action space expands from 6 to 9 actions. The model automatically adapts because `ActionSpace` dynamically reports sizes to the policy/value networks.

#### 6. Configuration (`utils/config.py`)

New defaults added:
```python
num_pad_multiples: int = 3
"""The number of pad multiple candidates (powers of 2: 2, 4, 8)."""

num_unroll_factors: int = 3
"""The number of unroll factor candidates (powers of 2: 2, 4, 8)."""
```

These are backward-compatible: existing configs without these fields use the defaults.

### Configuration

Example V4 config:

```json
{
  "implementation": "rl_autoschedular_v4",
  "num_pad_multiples": 3,
  "num_unroll_factors": 3
}
```

Notes:
- `num_pad_multiples` controls how many padding multiples the agent can choose from (default 3: multiples 2, 4, 8)
- `num_unroll_factors` controls how many unroll factors the agent can choose from (default 3: factors 2, 4, 8)
- These fields are optional; omitting them uses the defaults

### Files Modified

| File | Changes |
|------|---------|
| `rl_autoschedular_v4/transforms.py` | Added `transform_pad()`, `transform_pack()`, `transform_unroll()` |
| `rl_autoschedular_v4/actions/pad.py` | New `Pad` action class |
| `rl_autoschedular_v4/actions/pack.py` | New `Pack` action class |
| `rl_autoschedular_v4/actions/unroll.py` | New `Unroll` action class (terminal) |
| `rl_autoschedular_v4/actions/__init__.py` | Registered Pad, Pack, Unroll in `supported_actions` |
| `utils/config.py` | Added `num_pad_multiples` and `num_unroll_factors` defaults |
| `docs/VERSIONS.md` | Added V4 entry |
| `docs/Novelties/v4_action_space_expansion.md` | This document |
| `docs/NOVELTIES.md` | Updated Novelty 3 to reference V4 |

## How to Use

### Option 1: Default Expanded Action Space (Recommended)

```json
{
  "implementation": "rl_autoschedular_v4"
}
```

Uses default values for `num_pad_multiples` (3) and `num_unroll_factors` (3).

### Option 2: Custom Candidate Counts

```json
{
  "implementation": "rl_autoschedular_v4",
  "num_pad_multiples": 4,
  "num_unroll_factors": 4
}
```

Expands candidates to multiples/factors of 2, 4, 8, 16. Larger values increase action space dimensionality.

### Running the Pipeline

```bash
# Prepare base execution times
sbatch scripts/get_base.sh config/train1.json

# Split benchmark data
python scripts/split_json.py config/train1.json

# Train with expanded action space
sbatch scripts/train.sh config/train1.json

# Evaluate and compare
sbatch scripts/eval.sh config/train1.json

# View results in dashboard
streamlit run dashboard/dashboard.py --server.fileWatcherType none
```

All scripts automatically resolve `rl_autoschedular_v4` from the config.

## Expected Benefits

1. **Better Vectorization**: Pad enables power-of-two dimensions that vectorize more efficiently
2. **Improved Cache Locality**: Pack reorganizes data layout for tiled access patterns
3. **Higher ILP**: Unroll exposes more independent instructions to the backend
4. **Richer Schedule Space**: Agent can now compose tiling + padding + unrolling in a single schedule
5. **Composable Actions**: Pad and Pack are non-terminal, allowing sequences like `Tile → Pad → Vectorize`

## Validation Results

✅ Python compile checks passed for all new/modified files
✅ Import smoke test passed (`rl_autoschedular_v4.model`)
✅ `ActionSpace.size()` verified as 9 with correct symbols
✅ Pad transform tested on actual MLIR code (successful)
✅ Pack transform tested on actual MLIR code (successful)
✅ Unroll transform tested on actual MLIR code (successful)
✅ Model forward pass with expanded action space (successful)
✅ Action sampling distribution covers all 9 actions
✅ No baseline package import references inside V4 code

## Expected vs. Actual

### Hypothesis
- Expanded action space should enable schedules with better cache alignment and ILP
- Unroll should be particularly effective on compute-bound kernels with small loop bounds
- Pad should improve vectorization efficiency on non-power-of-two dimensions

### How to Validate
1. Train baseline (V0) and V4 on same benchmark suite
2. Compare final speedups achieved within same training budget
3. Analyze action frequency histograms:
   - Does V4 use Pad/Pack/Unroll meaningfully?
   - Are terminal actions (Unroll, Vectorization) chosen at appropriate points?
4. Compare generated schedules: V4 should produce longer, more diverse action sequences

### Metrics to Track
- **Final speedup**: Geometric mean across benchmark suite
- **Action diversity**: Entropy of action selection distribution
- **Schedule length**: Average number of steps before terminal action
- **Per-action success rate**: Fraction of Pad/Pack/Unroll applications that improve performance

## Limitations and Future Work

### Current Limitations
- **Pack access patterns are approximate**: `update_features` only updates loop bounds, not access matrices
- **Unroll is terminal**: Cannot apply further tag-based transforms after unrolling; this is a hard constraint from MLIR's transform dialect
- **Pad does not model padding overhead**: The agent does not see the cost of increased memory footprint from padding
- **No joint optimization**: Pad, Pack, and Unroll are chosen independently; no explicit coordination between them

### Future Improvements
- **Exact pack access tracking**: Update access matrices to reflect packed layout
- **Padding cost model**: Add padding overhead to shaped reward or observation
- **Non-terminal unroll**: Explore preserving tags through unrolling (may require MLIR dialect changes)
- **Joint action sampling**: Sample Pad + Pack parameters jointly instead of independently
- **Dynamic candidate sizes**: Adapt `num_pad_multiples` and `num_unroll_factors` based on loop bounds
- **Peel and Skew actions**: Add loop peeling and skewing as additional actions (originally proposed in thesis)

## Troubleshooting

### Pad Action Not Applying

Check that loop bounds are >= the chosen multiple. The action mask should prevent invalid choices, but if manually constructing actions, ensure multiples divide evenly or are <= bounds.

### Pack Action Fails

Pack requires that `packed_sizes` divide the loop bounds. The action mask enforces this, but if the mask is bypassed, the transform may fail. Check `Pack.action_mask()` logic.

### Unroll Not Allowed

Unroll is only allowed when `producer_tag is None`. If the agent is in a state where a producer has been selected for fusion, Unroll will be masked out. This is intentional: unrolling destroys the tag needed for fusion.

### Model Output Size Mismatch

If you see `RuntimeError: size mismatch` in the policy network, ensure `ActionSpace` reports the correct sizes:
```python
python -c "from rl_autoschedular_v4.actions import ActionSpace; print(ActionSpace.size(), ActionSpace.cumulative_mask_sizes())"
```
The model auto-adapts, but if checkpoints from an older version are loaded, sizes may mismatch.

## Ultimate-Version Assembly Notes

V4 was implemented with the same interface-preservation constraints as V1-V3:
- No training/evaluation script forks
- All new controls are namespaced under config keys (`num_pad_multiples`, `num_unroll_factors`)
- Policy/value model external call contracts are unchanged
- Action space expansion is fully dynamic

This allows combining V1 + V2 + V3 + V4 via composition without rewrites.

## References

- `docs/VERSIONS.md` - Version history and validation details
- `docs/NOVELTIES.md` - Thesis proposal mapping novelties to implementations
- `docs/Novelties/v1_hardware_aware_observation.md`
- `docs/Novelties/v2_shaped_reward.md`
- `docs/Novelties/v3_transformer_loop_nest_encoder.md`
- MLIR Transform Dialect: `mlir/include/mlir/Dialect/Transform/IR/TransformOps.td`
- `transform.structured.pad`, `transform.structured.pack`, `transform.loop.unroll` documentation in MLIR 19.x
