# MLIR-RL Configuration Details

This document explains all parameters available in the JSON configuration files used throughout the MLIR-RL project.

Configurations are governed by `utils.config.Config`, a singleton class that parses the target JSON configuration at its first import. The path to this JSON file must be specified using the `CONFIG_FILE_PATH` environment variable.

## File Organization & Loading

Configuration files are generally stored in `config/new_dataset/` and `config/old_dataset/`. When `Config()` is instantiated, it retrieves values from the JSON file, applying type checks and leveraging code-defined defaults if certain optional bounds are omitted. 

Below is an exhaustive list of all parameters, grouped by category:

---

## 1. Project & Execution Settings

These fields manage where artifacts are saved, which dataset is used, and which autoscheduler implementation is loaded.

* **`implementation`** (`str`, default: `"rl_autoschedular_v0"`): Auto-scheduler package implementation to use (e.g., `rl_autoschedular_v0`, `rl_autoschedular_v4_5`).
* **`benchmarks_folder_path`** (`str`): Path to the folder containing the input MLIR benchmarks. Can be empty if optimization mode is set to `"last"`.
* **`json_file`** (`str`, default: `""`): Path to the baseline JSON file tracking execution times for training benchmarks. If empty, the system derives it automatically via `utils.implementation.get_split_file_path`.
* **`eval_json_file`** (`str`, default: `""`): Path to the baseline JSON for evaluation benchmarks. Also auto-derived if left empty.
* **`results_dir`** (`str`): Base path to the directory where models, metrics, and JSON logs are saved for an experiment.
* **`main_exec_data_file`** (`str`): Path to the file storing/caching raw execution data.
* **`tags`** (`list[str]`): List of experiment tags to attach to the Neptune run (if logged).
* **`debug`** (`bool`): Flag to toggle verbose output or bypass certain crash handlers for debugging.

---

## 2. Action Space & Loop Nest Constraints

Configures heuristics and maximum boundaries for scheduling actions (like tiling, unrolling, and vectorization).

* **`truncate`** (`int`): The maximum number of scheduling steps allowed in a single episode.
* **`order`** (`list[list[str]]`): Defines the sequence of valid MLIR loop transformation actions (e.g., `["I"]`, `["!", "I", "NT"]`).
* **`max_num_loops`** (`int`): Maximum allowable depth of nested loops in a benchmark.
* **`max_num_stores_loads`** (`int`): Maximum acceptable number of load/store operations within a nested loop sequence.
* **`max_num_load_store_dim`** (`int`): Maximum dimension depth of buffers involved in loads/stores.
* **`num_tile_sizes`** (`int`): Number of valid tile size candidates available for the Tiling action.
* **`num_pad_multiples`** (`int`, default: `3`): Number of pad multiple candidates (typically powers of 2, e.g., 2, 4, 8).
* **`num_unroll_factors`** (`int`, default: `3`): Number of loop unroll factor candidates (typically powers of 2, e.g., 2, 4, 8).
* **`vect_size_limit`** (`int`): Upper bound on vector sizes to prevent the agent from applying unrealistically large vectorization targets.
* **`interchange_mode`** (`Enum ['enumerate', 'pointers', 'continuous']`): Action mechanism used to navigate and enforce loop interchange.

---

## 3. RL Training & Hyperparameters

Base training pipeline, batching, PPO (Proximal Policy Optimization) settings, and value/actor loss coefficients.

* **`nb_iterations`** (`int`): Total number of iterations comprising the training loop.
* **`bench_count`** (`int`): Number of batches evaluated within a single step/trajectory.
* **`lr`** (`float`): The global learning rate used for optimizer steps.
* **`ppo_epochs`** (`int`): Number of PPO training epochs over the current trajectory samples.
* **`ppo_batch_size`** (`Optional[int]`): Batch size corresponding directly to PPO updates.
* **`value_epochs`** (`int`): Number of value loss optimization epochs.
* **`value_batch_size`** (`Optional[int]`): Batch size tied to the value update steps.
* **`value_coef`** (`float`): Scalar coefficient scaling the magnitude of the value loss.
* **`value_clip`** (`bool`): Whether value clipping is enforced in the logic of the value loss to prevent abrupt network degradation.
* **`entropy_coef`** (`float`): Regularization coefficient promoting exploration by keeping policy prediction entropy uniformly spread.

### Advanced RL & Normalization
* **`exploration`** (`list['entropy', 'epsilon']`): Methods used to incentivize exploration. 
* **`init_epsilon`** (`float`): The starting configuration for epsilon-greedy rollouts.
* **`normalize_bounds`** (`Enum ['none', 'max', 'log']`): Configures whether upper bounds observed in the action space are normalized.
* **`normalize_adv`** (`Enum ['none', 'standard', 'max-abs']`): Method used to normalize PPO advantage estimations.
* **`reuse_experience`** (`Enum ['none', 'random', 'topk']`): Strategy to retrieve cached prior transitions for experience replay.
* **`replay_count`** (`int`): Trajectories maintained locally in the replay buffer.

---

## 4. Hardware Awareness

Hardware modeling settings, primarily used beginning with `"rl_autoschedular_v4.5"`, governing the hardware metrics passed into the model.

* **`hardware_auto_detect`** (`bool`, default: `True`): Toggles whether hardware metadata parameters below are auto-probed or forcibly set via explicit config. 
  *(The options below default at `0` indicating unknown prior logic, heavily reliant on `hardware_auto_detect` handling.)*
* **`hardware_l1_kb`** (`float`): L1 cache memory in KB.
* **`hardware_l2_kb`** (`float`): L2 cache memory in KB.
* **`hardware_l3_kb`** (`float`): L3 cache memory in KB.
* **`hardware_physical_cores`** (`int`): Absolute count of physical hardware CPU cores.
* **`hardware_logical_cores`** (`int`): Total logical CPU cores visible to the threads.
* **`hardware_simd_width`** (`int`): Width in bits of the processor SIMD unit (e.g., 256 for AVX2).
* **`hardware_clock_mhz`** (`float`): Native CPU Clock rate explicitly measured in MHz.

---

## 5. Reward Shaping

Interim heuristic scoring elements implemented to stabilize scheduling signals inside RL versions leveraging dense reward shaping strategies.

* **`reward_shaping_enabled`** (`bool`, default: `True`): Toggles dense intermediate heuristic signals vs. strict empirical execution times.
* **`reward_shaping_scale`** (`float`, default: `1.0`): Multiplier scaling the total impact of intermediate shaped rewards.
* **`reward_shaping_clip`** (`float`, default: `2.0`): The ceiling applied to clip individual reward shaping scores per term.
* **`reward_shaping_weight_ai`** (`float`, default: `1.0`): Magnitude of impact the Arithmetic Intensity calculation assigns to the final result.
* **`reward_shaping_weight_vectorizable`** (`float`, default: `0.1`): Relative weight promoting structural layout adjustments enabling vectorization.
* **`reward_shaping_weight_parallel`** (`float`, default: `0.1`): Relative weight driving actions to generate loop-parallel outputs.
* **`reward_shaping_vectorization_bonus`** (`float`, default: `0.2`): Concrete bonus multiplier implicitly handed over upon executing explicit vectorization steps.

---

## 6. Transformer Architecture

Underpinning the `"rl_autoschedular_v4_5"` versions, controls dimensions of loop nest encoders and self-attention layouts parsing MLIR graph operations.

* **`transformer_d_model`** (`int`, default: `256`): Dimensions matching token block embeddings entering the input model.
* **`transformer_nhead`** (`int`, default: `8`): Self-attention sequence parallel heads.
* **`transformer_num_layers`** (`int`, default: `3`): Stacking count for transformer encoder layers.
* **`transformer_ffn_dim`** (`int`, default: `1024`): Expanded intermediate tensor channel inside the feed-forward mechanism.
* **`transformer_dropout`** (`float`, default: `0.1`): Standard scaling dropout percentage limiting strict graph memorization.
* **`transformer_activation`** (`Enum ['relu', 'gelu']`, default: `'gelu'`): Internal gating strategy resolving nonlinear functions.
* **`transformer_pooling`** (`Enum ['cls', 'mean']`, default: `'cls'`): Pooling methodology mapping dynamic length seq outputs logically to fixed context windows.
* **`transformer_use_action_history_token`** (`bool`, default: `False`): Inject sequence context embedding strictly by action-tracing token rather than simple concatenated feed.

---

## 7. Evaluation Configuration

* **`eval_runs`** (`int`, default: `1`): Iterations run executing individual checks dynamically. Exposes evaluation sturdiness vs outliers.
* **`eval_aggregation`** (`Enum ['min', 'median', 'mean']`, default: `'min'`): Mathematical aggregation function merging iterations into consistent time benchmarks representing eval accuracy.
