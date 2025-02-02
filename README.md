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