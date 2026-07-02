## Getting Started
This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.
### Prerequisites:
###### Required
1) [CMake](https://cmake.org/): version 3.20 or greater.
2) [Ninja](https://ninja-build.org/).
3) [Gcc](https://gcc.gnu.org/) : version 13.2.
4) [Gxx]: version 13.2.
5) [LLD](https://lld.llvm.org/).
6) [Python](https://www.python.org/downloads/): version 3.11 or greater.
### Setup
#### 1. Building MLIR :
```sh
git clone --depth 1 -b release/19.x https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
-DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_ASSERTIONS=ON \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DLLVM_ENABLE_LLD=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON

cmake --build . --target check-mlir
```
#### 2. Install python requirements :
```sh
pip install -r requirements.txt
```
#### 3. Setup environment variables :
Change llvm related variables according to your llvm-project folder path.
```env
NEPTUNE_PROJECT=<NEPTUNE_PROJECT_URL>
NEPTUNE_TOKEN=<NEPTUNE_API_TOKEN>
LLVM_BUILD_PATH=llvm-project/build
MLIR_SHARED_LIBS=llvm-project/build/lib/libomp.so,llvm-project/build/lib/libmlir_c_runner_utils.so,llvm-project/build/lib/libmlir_runner_utils.so
AST_DUMPER_BIN_PATH=tools/ast_dumper/build/bin/AstDumper
VECTORIZER_BIN_PATH=tools/vectorizer/build/bin/Vectorizer
```
### Documentation
#### 1. Jobs
For running jobs using slurm script examples are provided in the `scripts/` folder.
#### 2. Configuration
Configuring the model on a specific case can be done by setting a JSON config file containing all required settings. Configuration JSON file examples are provided in the `config/` folder.
The following JSON content is an example of a config file:
```json
{
    "max_num_stores_loads": 7,
    "max_num_loops": 7,
    "max_num_load_store_dim": 7,
    "num_tile_sizes": 7,
    "num_transformations": 6,
    "vect_size_limit": 2048,
    "use_bindings": false,
    "use_vectorizer": false,
    "data_format": "json",
    "optimization_mode": "last",
    "benchmarks_folder_path": "",
    "len_trajectory": 64,
    "ppo_batch_size": 64,
    "nb_iterations": 10000,
    "ppo_epochs": 4,
    "entropy_coef": 0.01,
    "lr": 0.001,
    "truncate": 5,
    "json_file": "data/nn/train_operations.json",
    "tags": ["nn"],
    "logging": true
}
```
The following list describes every required setting in a configuration file.
- `max_num_stores_loads (int)`: The maximum number of loads in the nested loops.
- `max_num_loops (int)`: The max number of nested loops.
- `max_num_load_store_dim (int)`: The max number of dimensions in load/store buffers.
- `num_tile_sizes (int)`: The number of possible tile sizes for a loop.
- `num_transformations (int)`: The number of transformations.
- `vect_size_limit (int)`: Vectorization size limit to prevent large sizes vectorization.
- `use_bindings (bool)`: Flag to enable using python bindings for execution, if False, the execution will be done using the command line. Default is False.
- `use_vectorizer (bool)`: Flag to enable using the vectorizer C++ program for vectorization, if False, vectorization is done using transform dialect directly. Default is False.
- `data_format (Literal["json", "mlir"])`: The format of the data, can be either "json" or "mlir". "json" mode reads json files containing benchmark features, "mlir" mode reads mlir code files directly and extract features from it using AST dumper. Default is "json".
- `optimization_mode (Literal["last", "all"])`: The optimization mode to use, "last" will optimize only the last operation, "all" will optimize all operations in the code. Default is "last".
- `benchmarks_folder_path (str)`: Path to the benchmarks folder. Can be empty if data format is set to "json".
- `len_trajectory (int)`: Length of the trajectory used for PPO.
- `ppo_batch_size (int)`: Batch size for PPO.
- `nb_iterations (int)`: Number of training iterations.
- `ppo_epochs (int)`: Number of epochs for PPO.
- `entropy_coef (float)`: Entropy coefficient.
- `lr (float)`: Learning rate.
- `truncate (int)`: Maximum number of steps of a schedule for an operation.
- `json_file (str)`: Path to the JSON file containing the benchmarks code and features if data format is set to "json". Otherwise, it should contain original execution times for every benchmark in the benchmark folder.
- `tags (list[str])`: List of tags to add to the neptune experiment.
- `logging (bool)`: Flag to enable logging to neptune.

Additional fields for versioned agents:
- `implementation (str)`: Autoscheduler package to run (for example `rl_autoschedular`, `rl_autoschedular_v1`, `rl_autoschedular_v2`, `rl_autoschedular_v4_5`).
- `results_dir (str)`: Root directory for all experiment outputs (e.g., `results/experiment3`).
- `hardware_auto_detect (bool)`: If true, hardware features are auto-detected from the current machine (recommended for single-HPC training/eval).
- `hardware_l1_kb`, `hardware_l2_kb`, `hardware_l3_kb`, `hardware_physical_cores`, `hardware_logical_cores`, `hardware_simd_width`, `hardware_clock_mhz`: Optional manual overrides for advanced cross-hardware experiments. For single-machine runs, leave these unset to avoid host/override mismatch.
- `reward_shaping_enabled (bool)`: Enables dense intermediate reward shaping for shaped-reward agents.
- `reward_shaping_scale (float)`: Global multiplier for shaped reward delta.
- `reward_shaping_clip (float)`: Clip bound for each shaped reward term.
- `reward_shaping_weight_ai (float)`: Weight for arithmetic-intensity proxy in shaped reward.
- `reward_shaping_weight_vectorizable (float)`: Weight for vectorizability score in shaped reward.
- `reward_shaping_weight_parallel (float)`: Weight for parallel-loop ratio in shaped reward.
- `reward_shaping_vectorization_bonus (float)`: Extra bonus for explicit vectorization actions.
- `transformer_d_model (int)`: Hidden size for transformer token embeddings used by transformer-based agents.
- `transformer_nhead (int)`: Number of attention heads in transformer encoder layers.
- `transformer_num_layers (int)`: Number of transformer encoder layers.
- `transformer_ffn_dim (int)`: Feed-forward hidden dimension used inside each transformer encoder layer.
- `transformer_dropout (float)`: Dropout ratio used in transformer projections and encoder blocks.
- `transformer_activation (Literal["relu", "gelu"])`: Activation function used in transformer feed-forward layers.
- `transformer_pooling (Literal["cls", "mean"])`: Pooling strategy for the transformer sequence output.
- `transformer_use_action_history_token (bool)`: If true, action history is encoded as a transformer token instead of concatenated after pooling.

---

## Troubleshooting an Inherited LLVM Build

If the `llvm-project/build/` directory was compiled by another user, the MLIR Python bindings will be broken because all `.py` files in the build are symlinks pointing to the original user's scratch path. Follow these steps to fix the environment.

### Problem 1: `GLIBCXX_3.4.29` not found

The system `libstdc++` may be too old for numpy/torch. Fix by pointing to the venv's newer library:

```sh
export LD_LIBRARY_PATH=~/envs/mlir/lib:$LD_LIBRARY_PATH
```

### Problem 2: Broken MLIR Python symlinks

All `.py` files under `build/tools/mlir/python_packages/mlir_core/` are symlinks to the original builder's path. Since the git repo tracks the source files but they may not be checked out, restore them first:

```sh
cd llvm-project

# Restore source Python files from git
git checkout HEAD -- $(git ls-files mlir/python/mlir/ | tr '\n' ' ')
```

Then rewrite all broken symlinks to use your own path (replace `OTHER_USER` and `YOUR_USER`):

```sh
find build/tools/mlir/python_packages/mlir_core -type l | while read link; do
    target=$(readlink "$link")
    if echo "$target" | grep -q "OTHER_USER"; then
        new_target=$(echo "$target" | sed 's|/scratch/OTHER_USER/|/scratch/YOUR_USER/|g')
        rm "$link" && ln -s "$new_target" "$link"
    fi
done
```

> **Note:** Use `rm` + `ln -s` rather than `ln -sf` — the `-f` flag can fail with "Permission denied" on broken symlinks to inaccessible paths on some filesystems, even when you own the symlink.

### Problem 3: Missing environment variables

Add the following to your virtualenv's `activate` script (`~/envs/mlir/bin/activate`) so they are set automatically on every activation:

```sh
# MLIR Python bindings
export PYTHONPATH=/scratch/YOUR_USER/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=~/envs/mlir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MLIR_SHARED_LIBS=/scratch/YOUR_USER/MLIR-RL/llvm-project/build/lib/libmlir_runner_utils.so,/scratch/YOUR_USER/MLIR-RL/llvm-project/build/lib/libmlir_c_runner_utils.so
```

Verify the fix with:

```sh
python -c "from mlir.ir import Context; print('OK')"
```

### Problem 4: V4 Reliability and "experiment3" Transition

V4 (Integrated) initially suffered from high failure rates due to aggressive incentives. **V4.5 (Robust Integration)** fixes this with:
- **Process Isolation:** Mandatory subprocess execution for transforms and execution.
- **Success-Contingent Rewards:** Negating shaped rewards on final failure.
- **experiment3:** Standardized results directory to avoid mixing legacy artifacts.

Verify V4.5 functionality:
```bash
export CONFIG_FILE_PATH=config/v4_5.json
python -c "from rl_autoschedular_v4_5.execution import execute_code; print('V4.5 Load OK')"
```

### Generating base execution times

Once the environment is set up, generate base execution times for training:

```sh
python scripts/baseline/get_base.py --benchmarks-dir data/nn/code_files --output data/nn/base_exec_times.json
```