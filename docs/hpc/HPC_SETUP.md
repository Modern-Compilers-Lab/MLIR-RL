# New Installation (HPC)

# Local Installation

- Notion Guide : [https://yielding-sheet-a7e.notion.site/Installation-Running-MLIR-RL-Agent-On-Ubuntu-24-24dc22bb1876802e94dedbf2c0223777](https://www.notion.so/24dc22bb1876802e94dedbf2c0223777?pvs=21)

# Conda Environment Setup

### Create conda environment

Install needed packages

```bash
conda create -n mlir
conda activate mlir
```

### HPC GCC

```bash
conda install -c -y conda-forge python=3.11 cmake ninja numpy PyYAML pybind11
which gcc
which g++
which c++
```

### GCC (did not work, faced recurring linking issues)

```bash
conda install -c -y conda-forge python=3.11 gxx_linux-64=13.2.0 gcc_linux-64=13.2.0 libstdcxx-ng libgcc-devel_linux-64 cmake ninja numpy PyYAML pybind11
```

```bash
# set these environment variables BEFORE running cmake
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
export CXXFLAGS="-I$CONDA_PREFIX/include"
export CFLAGS="-I$CONDA_PREFIX/include"

# most importantly - force atomic library
export LDFLAGS="$LDFLAGS -latomic"
```

### Create `gcc` and `g++` symbolic links

```bash
cd ~/.conda/envs/mlir/bin/
ln -s x86_64-conda-linux-gnu-gcc gcc
ln -s x86_64-conda-linux-gnu-g++ g++
ln -s x86_64-conda-linux-gnu-ar ar
```

### Verify the symbolic links and compilation

```bash
# verify
which gcc
which g++
gcc --version
g++ --version

# test compilation
echo '#include <cstddef>' | g++ -x c++ -c - -o /dev/null
echo '#include <cstdlib>' | g++ -x c++ -c - -o /dev/null
```

### Clang (recommended for LLVM 19.x)

Use clang from conda-forge. GCC 13.x on RHEL8 HPC nodes had recurring linking issues; clang works reliably.

```bash
conda install -c -y conda-forge python=3.11 clang clangxx lld cmake ninja numpy PyYAML pybind11
cmake --version
clang --version
clang++ --version
```

> **Note:** The current conda-forge clang is version 21.x. LLVM 19.x source has a compatibility issue with clang 21 where `int64_t` / `uint32_t` are no longer transitively included. The fix is documented in the cmake section below.

# LLVM Installation

### Clone MLIR-RL repo

```bash
cd $SCRATCH
git clone https://github.com/Modern-Compilers-Lab/MLIR-RL.git
```

### Clone LLVM repo inside MLIR-RL

```bash
cd MLIR-RL
git clone --depth 1 -b release/19.x https://github.com/llvm/llvm-project.git
```

### HPC libstdc++ prerequisite

The RHEL8 cluster has an old system `libstdc++` that is incompatible with numpy/torch. The cluster provides a newer GCC 14 via spack:

```bash
export GCC14_LIB=/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-14.2.0-wfwb3ds4a5thcsh5w5o23k6wq7ob5ok3/lib64
export LD_LIBRARY_PATH=$GCC14_LIB:$LD_LIBRARY_PATH
```

Find this path for your cluster with:
```bash
ls /share/apps/NYUAD6/spack/*/opt/spack/linux-rocky8-zen/gcc-*/gcc-14.*/lib64 2>/dev/null
```

### Build LLVM with MLIR and Python Bindings (Working Command)

```bash
mkdir -p llvm-project/build

cmake -S llvm-project/llvm \
      -B llvm-project/build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_BUILD_EXAMPLES=ON \
      -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/clang \
      -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/clang++ \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DPython3_EXECUTABLE=$(which python3) \
      -DCMAKE_CXX_FLAGS="-include cstdint" \
      -DCMAKE_C_FLAGS="-include stdint.h" \
      -DCMAKE_EXE_LINKER_FLAGS="-L$GCC14_LIB -Wl,-rpath,$GCC14_LIB" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L$GCC14_LIB -Wl,-rpath,$GCC14_LIB"
```

**Key fixes explained:**

| Flag | Reason |
|------|--------|
| `-DLLVM_ENABLE_PROJECTS=mlir` | Only MLIR is needed for this project (skip clang/openmp for faster build) |
| `-DCMAKE_CXX_FLAGS="-include cstdint"` | **Clang 21 compatibility:** LLVM 19.x headers use `int64_t`/`uint32_t` without `#include <cstdint>`. Clang 21 no longer transitively includes these types. This flag forces the include globally. |
| `-DCMAKE_C_FLAGS="-include stdint.h"` | Same fix for C compilation units. |
| `-DCMAKE_EXE_LINKER_FLAGS="-L$GCC14_LIB -Wl,-rpath,$GCC14_LIB"` | Link against the cluster's GCC 14 libstdc++ at both link-time and runtime. Required for numpy/torch compatibility. |

### Build

```bash
ninja -C llvm-project/build -j$(nproc)
```

> **Note:** Building on 18 cores takes ~1-2 hours. The output is ~8-10 GB in the build directory.

### Add MLIR binaries and Python bindings to `PATH`

```bash
export PATH=/scratch/<NetID>/MLIR-RL/llvm-project/build/bin:$PATH
export PYTHONPATH=/scratch/<NetID>/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH
```

### Test MLIR and Python bindings

```bash
mlir-opt --version
python3 -c "import mlir; print('MLIR Python bindings loaded successfully')"
```

```python
python3 << 'EOF'
from mlir.ir import Context, Module

with Context() as ctx:
    module = Module.parse("""
    module {
      func.func @add(%arg0: i32, %arg1: i32) -> i32 {
        %0 = arith.addi %arg0, %arg1 : i32
        return %0 : i32
      }
    }
    """)
    print(module)
    print("\nMLIR working correctly!")
EOF
```

### To make these paths permanent, add to your `~/.bashrc` :

```bash
cat >> ~/.bashrc << 'EOF'

# MLIR paths for MLIR-RL project
export PATH=/scratch/<NetID>/MLIR-RL/llvm-project/build/bin:$PATH
export PYTHONPATH=/scratch/<NetID>/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH
EOF
```

```bash
source ~/.bashrc
```

# 8. Tools Installation

Within **MLIR-RL** repository execute : 

```bash
cd $SCRATCH/MLIR-RL
PROJECT_ROOT=$(pwd)
```

## AST Dumper

```bash
cd tools/ast_dumper
mkdir build
cd build
```

### Adjust the paths accordingly for LLLVM and MLIR

```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=$SCRATCH/MLIR-RL/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=$SCRATCH/MLIR-RL/llvm-project/build/lib/cmake/mlir \
  ..
  
cmake --build .
```

## Vectorizer

```bash
cd $SCRATCH/MLIR-RL
cd tools/vectorizer
mkdir build
cd build
```

```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=$SCRATCH/MLIR-RL/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=$SCRATCH/MLIR-RL/llvm-project/build/lib/cmake/mlir \
  ..
  
cmake --build .
```

# 9. Install Python Packages

Make sure your Conda environment is activated

```bash
cd $SCRATCH/MLIR-RL
python -m pip install -r requirements.txt
```

# 10. Create a Neptune Project

[neptune.ai | Experiment tracker purpose-built for foundation models](https://neptune.ai/)

# 11. Setup Environment Variables

Set `.env` (looking like the following)

```bash
CONDA_ENV=/home/<NetID>/envs/mlir/bin/activate
NEPTUNE_PROJECT=<Neptune Project>
NEPTUNE_API_TOKEN=<Neptune API Token>
LLVM_BUILD_PATH=/scratch/<NetID>/MLIR-RL/llvm-project/build
MLIR_SHARED_LIBS=/home/<NetID>/envs/mlir/lib/libomp.so,/scratch/<NetID>/MLIR-RL/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/<NetID>/MLIR-RL/llvm-project/build/lib/libmlir_runner_utils.so
AST_DUMPER_BIN_PATH=/scratch/<NetID>/MLIR-RL/tools/ast_dumper/build/bin/AstDumper
VECTORIZER_BIN_PATH=/scratch/<NetID>/MLIR-RL/tools/vectorizer/build/bin/Vectorizer

# HPC cluster: newer libstdc++ required
GCC14_LIB=/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-14.2.0-wfwb3ds4a5thcsh5w5o23k6wq7ob5ok3/lib64
LD_LIBRARY_PATH=$GCC14_LIB
PATH=/home/<NetID>/envs/mlir/bin:$PATH

export LD_LIBRARY_PATH=/home/<NetID>/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/scratch/<NetID>/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH
MAIL_USER=<your email>
```

### Working .env example (mb10856):

```bash
# CONDA_ENV=/home/mb10856/.conda/envs/mlir
CONDA_ENV=/home/mb10856/envs/mlir/bin/activate
NEPTUNE_PROJECT=<Neptune Project>
NEPTUNE_API_TOKEN=<Neptune API Token>
LLVM_BUILD_PATH=/scratch/mb10856/MLIR-RL/llvm-project/build
MLIR_SHARED_LIBS=/home/mb10856/envs/mlir/lib/libomp.so,/scratch/mb10856/MLIR-RL/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/mb10856/MLIR-RL/llvm-project/build/lib/libmlir_runner_utils.so
AST_DUMPER_BIN_PATH=/scratch/mb10856/MLIR-RL/tools/ast_dumper/build/bin/AstDumper
VECTORIZER_BIN_PATH=/scratch/mb10856/MLIR-RL/tools/vectorizer/build/bin/Vectorizer

# The mlir conda env requires a newer libstdc++ on this cluster:
GCC14_LIB=/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-14.2.0-wfwb3ds4a5thcsh5w5o23k6wq7ob5ok3/lib64
LD_LIBRARY_PATH=$GCC14_LIB
PATH=/home/mb10856/envs/mlir/bin:$PATH

export LD_LIBRARY_PATH=/home/mb10856/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/scratch/mb10856/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH
MAIL_USER=<your email>
```

# 12. Change the config file

In config/config.json put this :

```json
{
    "max_num_stores_loads": 7,
    "max_num_loops": 12,
    "max_num_load_store_dim": 12,
    "num_tile_sizes": 7,
    "vect_size_limit": 512,
    "order": [["I"], ["!", "I", "NT"], ["!", "I"], ["V", "NT"]],
    "interchange_mode": "pointers",
    "exploration": ["entropy"],
    "init_epsilon": 0.5,
    "new_architecture": false,
    "normalize_bounds": "max",
    "normalize_adv": "standard",
    "sparse_reward": true,
    "split_ops": true,
    "reuse_experience": "none",
    "activation": "relu",
    "benchmarks_folder_path": "data/all/code_files",
    "bench_count": 64,
    "replay_count": 10,
    "nb_iterations": 10000,
    "ppo_epochs": 4,
    "ppo_batch_size": 32,
    "value_epochs": 0,
    "value_batch_size": 32,
    "value_coef": 0.5,
    "value_clip": false,
    "entropy_coef": 0.01,
    "lr": 0.001,
    "truncate": 5,
    "json_file": "data/all/execution_times_train.json",
    "eval_json_file": "data/all/execution_times_eval.json",
    "tags": ["all", "pointers", "IPFT~V", "entropy", "standard", "dask", "new-v"],
    "debug": false,
    "main_exec_data_file": "",
    "results_dir": "results"
}
```

## Troubleshooting

### Clang 21 + LLVM 19.x: missing int64_t / uint32_t

If you see errors like:
```
error: unknown type name 'int64_t'
error: use of undeclared identifier 'uint32_t'
```

This is because clang 21 no longer transitively includes `<cstdint>` types. The fix is already included in the cmake command above via:
```bash
-DCMAKE_CXX_FLAGS="-include cstdint" \
-DCMAKE_C_FLAGS="-include stdint.h"
```

### Inherited build from another user

If you inherit an `llvm-project/build/` compiled by another user (with their NetID in CMakeCache.txt), do **not** try to fix the symlinks — wipe and rebuild from scratch:

```bash
rm -rf llvm-project/build
# Then re-run the full cmake + ninja procedure above
```

The old build has stale paths in CMakeCache.txt pointing to another user's home/scratch, which will cause linking and runtime path resolution failures.

### GLIBCXX version errors at runtime

If you see `GLIBCXX_3.4.29 not found`, make sure the GCC14 lib path is in `LD_LIBRARY_PATH`:

```bash
export GCC14_LIB=/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-14.2.0-wfwb3ds4a5thcsh5w5o23k6wq7ob5ok3/lib64
export LD_LIBRARY_PATH=$GCC14_LIB:$LD_LIBRARY_PATH
```

### Python bindings import fails

Verify the PYTHONPATH includes the build output:
```bash
export PYTHONPATH=/scratch/<NetID>/MLIR-RL/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH
python3 -c "from mlir.ir import Context; print('OK')"
```