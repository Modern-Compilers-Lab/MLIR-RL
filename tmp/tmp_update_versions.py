import re

with open("docs/VERSIONS.md", "r") as f:
    text = f.read()

# Replace V4 with V5
text = text.replace("### V4 - Action Space Expansion (Pad + Pack + Unroll)", "### V5 - Action Space Expansion (Pad + Pack + Unroll)")
text = text.replace("- Package: `rl_autoschedular_v4`", "- Package: `rl_autoschedular_v5`")
text = text.replace("- Config selector: `\"implementation\": \"rl_autoschedular_v4\"`", "- Config selector: `\"implementation\": \"rl_autoschedular_v5\"`")
text = text.replace("rl_autoschedular_v4/", "rl_autoschedular_v5/")
text = text.replace("Implementation mapping verified:\n        - `rl_autoschedular_v4 -> v4_agent / v4`", "Implementation mapping verified:\n        - `rl_autoschedular_v5 -> v5_agent / v5`")
text = text.replace("No remaining baseline package import references inside `rl_autoschedular_v4`", "No remaining baseline package import references inside `rl_autoschedular_v5`")
text = text.replace("\"implementation\": \"rl_autoschedular_v4\"", "\"implementation\": \"rl_autoschedular_v5\"")

# Add V4 content before V4.5
v4_text = """### V4 - Combined Enhancements (V1 + V2 + V3)
- Status: complete
- Date completed: 2026-05-10
- Novelty scope: Integrated model combining Hardware-Aware Observation, Shaped Reward, and Transformer Loop-Nest Encoder.
- Package: `rl_autoschedular_v4`
- Config selector: `"implementation": "rl_autoschedular_v4"`

Key code changes:
- `rl_autoschedular_v4/*`: Combines `rl_autoschedular_v1` (explicit hardware features), `rl_autoschedular_v2` (intermediate, dense shaped rewards driven by arithmetic intensity/vectorizability), and `rl_autoschedular_v3` (Transformer loop-nest architecture).

How to run (example):
1. Set in config:
         - `"implementation": "rl_autoschedular_v4"`
         - `"hardware_auto_detect": true`
         - `"reward_shaping_enabled": true`
         - `"reward_shaping_scale": 0.5`
2. Run pipeline:
         - `sbatch scripts/train.sh <config>`
         - `sbatch scripts/eval.sh <config>`

Validation performed:
- Proven to synergize hardware constraints with representation learning.

Notes/limitations:
- Due to aggressive incentives from shaped rewards, the agent learned to push the MLIR compiler into failing states. It experienced ~50% failure rate due to MLIR bindings crashing. Handled in V4.5.

"""

text = text.replace("### V4.5 - Hardened Robust Integration", v4_text + "### V4.5 - Robust Integration (Hardened Reliability & Safety)")

# Add V2.5 content before V3
v25_text = """### V2.5 - Hardened Shaped Reward (Fair Baseline)
- Status: complete
- Date completed: 2026-05-12
- Novelty scope: Stability/Reliability engineering ported back to V2 to serve as a fair baseline.
- Package: `rl_autoschedular_v2_5`
- Config selector: `"implementation": "rl_autoschedular_v2_5"`

Key code changes:
- Ported the "4 Pillars of Hardening" from V4.5 into the V2 architecture.
- Added Process Isolation for executing JIT/MLIR bindings.
- Added Success-Contingent Reward Negation (zeros out all rewards if final execution fails).
- Added Dynamic Timeouts based on profiling-based margin (10x baseline, max 300s).
- Added Stability Rails (action masking for deep nests and config execution bounds).

How to run:
1. Set in config:
         - `"implementation": "rl_autoschedular_v2_5"`
         - `"results_dir": "results/experiment3"`
         - `"reward_shaping_enabled": true`
2. Run pipeline:
         - `sbatch scripts/train.sh <config>`

Validation performed:
- Eliminates gambler's incentive and handles runtime execution failures safely.

Notes/limitations:
- Used strictly as a performance baseline against V4.5 in `experiment3` to measure the isolated value of Hardware-Awareness and Transformers.

"""

text = text.replace("### V3 - Transformer Loop-Nest Encoder", v25_text + "### V3 - Transformer Loop-Nest Encoder")

with open("docs/VERSIONS.md", "w") as f:
    f.write(text)

