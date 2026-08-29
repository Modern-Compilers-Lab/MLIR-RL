# `iql/` — Offline RL with Implicit Q-Learning (Phase 4)

**Phase 4 of the [MLIR-RL](../README.md) research programme. Research prototype — see
[Status](#status--read-before-you-run-anything) before running anything.**

## Why offline RL here

Every step of the online RL loop in [../rl_autoschedular/](../rl_autoschedular/) costs a **real
compile and execute** — apply the Transform-dialect schedule, bufferize, lower to LLVM, JIT, run,
time. That is seconds to minutes per step, and PPO is on-policy: each rollout is consumed by a
handful of gradient updates and then thrown away.

Offline RL breaks that coupling. Collect a dataset of trajectories once (random rollouts, earlier
policies, expert schedules), then run as many gradient updates over it as you like without touching
the compiler. If it works, the same compute budget buys far more learning.

## Why IQL specifically

The usual failure mode of offline RL is **extrapolation error**: the Bellman backup queries
`Q(s', a')` for actions the dataset never contains, the value function over-estimates them, and the
error compounds. Implicit Q-Learning ([Kostrikov et al., 2021](https://arxiv.org/abs/2110.06169))
sidesteps this by never querying an out-of-distribution action:

1. **Expectile value regression.** Fit `V(s)` to the `tau`-expectile of `Q(s,a)` over *dataset*
   actions. With `tau > 0.5` this approximates a maximum over the actions actually present, without
   evaluating any action outside the data.
2. **Twin Q with a target network.** `Q(s,a) ← r + gamma * V(s')` on two Q heads, taking the
   minimum, with a Polyak-averaged target (coefficient `alpha`) — the standard pessimism and
   stability machinery.
3. **Advantage-weighted policy extraction.** The policy is fit by weighted behaviour cloning with
   weights `exp(beta * (Q(s,a) - V(s)))` — it imitates the dataset's good actions, never proposes new
   ones during training.

This suits the compiler setting well: the action space is large and structured, most action/parameter
combinations are illegal or catastrophic, and a value function that freely extrapolates over them
would be worthless.

## Files

| File | Contents |
|---|---|
| [agent.py](agent.py) | `IQLAgent` — owns the three networks, their optimizers, the target Q, and the update rules. Deliberately aligned with the PPO architecture: shared 3×512 backbone and **hierarchical heads** (one action head + per-action parameter heads), so it consumes the same observations and the same action space |
| [value_function.py](value_function.py) | `IQLValueModel` — `obs → 512 → 512 → 512 → 1`, trained by expectile regression |
| [q_functions.py](q_functions.py) | `IQLTwinQ`, `_TwinHiearchicalQNetwork`, `_DiscreteQHead` — twin Q networks with the same hierarchical head layout |
| [policy.py](policy.py) | `IQLPolicyModel` — shared backbone plus one head per action-parameter group |
| [config.py](config.py) | A `Config` singleton (via [singleton.py](singleton.py)) loaded from the JSON file named by the **`OFFLINE_RL_CONFIG_FILE_PATH`** environment variable |
| [singleton.py](singleton.py) | Singleton metaclass |

Everything else is reused from phase 1: the environment
([../rl_autoschedular/env.py](../rl_autoschedular/env.py)), the action space
([../rl_autoschedular/actions/](../rl_autoschedular/actions/)), and the observation encoding
(`Observation.get_parts(obs, OpFeatures, ActionHistory)` from
[../rl_autoschedular/observation.py](../rl_autoschedular/observation.py)). The dataset container is
`OfflineDataset` in [../utils/data_collector.py](../utils/data_collector.py), which stores
`obs / actions / rewards / next_obs / dones` as an `.npz`.

Note the environment variable is **`OFFLINE_RL_CONFIG_FILE_PATH`**, deliberately distinct from the
PPO path's `CONFIG_FILE_PATH`.

## Configuration

Defaults from [config.py](config.py):

| Key | Default | Meaning |
|---|---|---|
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.7 | Expectile for value regression; `> 0.5` biases towards the max |
| `inverse_temperature` | 3.0 | The `beta` in the advantage-weighted policy loss |
| `alpha` | 0.005 | Polyak coefficient for the target Q |
| `batch_size` | 256 | |
| `learning_rate` | `{"value": 3e-4, "q": 3e-4, "policy": 3e-4}` | Per-network learning rates |
| `max_steps` | 1,000,000 | Gradient steps |
| `target_update_freq` | 1 | Target-network update cadence |
| `sparse_reward` | `true` | Reward only at the end of a schedule |
| `offline_data_directory` | `./data` | Where the dataset lives |
| `offline_data_file` | `offline_data.npz` | Dataset filename |

It also re-declares **every** phase-1 environment key — `max_num_loops`,
`max_num_stores_loads`, `max_num_load_store_dim`, `num_tile_sizes`, `vect_size_limit`, `order`,
`interchange_mode`, `exploration`, `init_epsilon`, `normalize_bounds`, `split_ops`, `activation`,
`benchmarks_folder_path`, `bench_count`, `truncate`, `json_file`, `eval_json_file`, `tags`, `debug`,
`exec_data_file`, `results_dir` — so an offline run is self-describing. Their meanings are documented in
[../rl_autoschedular/README.md](../rl_autoschedular/README.md#4-configuration).

## Entry points

Both live at the repository root and are run from there.

| Script | Purpose |
|---|---|
| [../train_iql_offline.py](../train_iql_offline.py) | Pure offline training. Loads `OfflineDataset` into tensors, then loops gradient updates; evaluates periodically against a live `Env` |
| [../train_iql_online.py](../train_iql_online.py) | Online fine-tuning warm-started from the offline buffer. The offline/online sampling ratio decays over `MAX_STEPS_OFFLINE_ONLINE_RATIO = 100_000` steps, with `UPDATE_ITERS = 3` gradient updates per collected batch |

Datasets are produced by rolling out action sequences with [../fill_db.py](../fill_db.py).

## Status — read before you run anything

**This is an unfinished prototype and does not run as-is on this branch.** It came in through a
branch merge (`cade55d Img2Col as an action; Offline RL (Ouail contribution); LLM Action space PoC`)
and was never brought back in sync with the refactored phase-1 code. Four concrete breaks:

1. **Wrong module name.** [../train_iql_online.py](../train_iql_online.py) does
   `from iql.iql_agent import IQLAgent`, but the module is [agent.py](agent.py). Either rename the
   file or fix the import.
2. **Wrong config class.** [agent.py](agent.py) does `from utils.config import Config` and then reads
   `cfg.gamma`, `cfg.tau`, `cfg.beta`, `cfg.alpha` and `cfg.lr["value"]`. But
   [../utils/config.py](../utils/config.py) has none of those and declares `lr` as a **float**, not a
   dict. It needs `iql.config.Config`.
3. **Key mismatch inside `iql/config.py`.** [agent.py](agent.py) reads `cfg.beta`, while
   [config.py](config.py) declares the same quantity as `inverse_temperature`. Pick one name.
4. **Dataset key mismatch.** Both trainers read `cfg.offline_data_save_dir`, while
   [config.py](config.py) declares `offline_data_directory`.

Additionally, both trainers import `from rl_autoschedular import config as cfg, file_logger as fl`,
which [../rl_autoschedular/__init__.py](../rl_autoschedular/__init__.py) does not export — the same
stale import that affects several other root scripts
([../rl_autoschedular/README.md](../rl_autoschedular/README.md#6-known-stale-entry-points)).

None of this is deep breakage: the networks and the IQL update rules in this directory are complete
and self-consistent. Reviving the direction means fixing the wiring above, regenerating an offline
dataset with `fill_db.py`, and validating that the offline-trained policy is competitive with the PPO
baseline on the same evaluation split.
