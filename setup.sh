#!/bin/bash

# Set project root
PROJECT_ROOT=$(pwd)
VENV_PATH="${PROJECT_ROOT}/mlir-venv"

# Step 0: Create and activate venv
python3.11 -m venv ${VENV_PATH}
source ${VENV_PATH}/bin/activate

# Upgrade pip in venv
pip install --upgrade pip

# Step 1: Install project Python requirements in venv
pip install -r requirements.txt

# Step 2: Clone and build MLIR in venv
git clone --depth 1 -b release/19.x https://github.com/llvm/llvm-project.git
cd llvm-project
sudo mkdir build
sudo cd build
sudo cmake -S ../llvm -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
-DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_ASSERTIONS=ON \
-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
-DPython3_EXECUTABLE=${VENV_PATH}/bin/python
sudo cmake --build . --target check-mlir
sudo cmake --build . --target check-mlir-python  # Extra test for bindings
cd ${PROJECT_ROOT}

# Step 2.1: In case OMP didn't build
cd llvm-project/build
sudo ninja omp


# Step 3: Install MLIR Python binding requirements in venv
cd llvm-project/mlir/python
sudo pip install -r requirements.txt  # This includes NumPy
cd ${PROJECT_ROOT}


# For MLIR specific project 

# Step 4: Build AstDumper if directory exists
if [ -d "tools/ast_dumper" ]; then
  cd tools/ast_dumper
  mkdir build
  cd build
  sudo cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=${PROJECT_ROOT}/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=${PROJECT_ROOT}/llvm-project/build/lib/cmake/mlir \
  -DPython3_EXECUTABLE=${VENV_PATH}/bin/python \
  ..
  sudo cmake --build .
  cd ${PROJECT_ROOT}
else
  echo "Warning: tools/ast_dumper not found. Skipping build. Ensure project repo is cloned correctly."
fi

# Step 5: Build Vectorizer if directory exists
if [ -d "tools/vectorizer" ]; then
  cd tools/vectorizer
  mkdir build
  cd build
  sudo cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=${PROJECT_ROOT}/llvm-project/build/lib/cmake/llvm \
  -DMLIR_DIR=${PROJECT_ROOT}/llvm-project/build/lib/cmake/mlir \
  -DPython3_EXECUTABLE=${VENV_PATH}/bin/python \
  ..
  sudo cmake --build .
  cd ${PROJECT_ROOT}
else
  echo "Warning: tools/vectorizer not found. Skipping build. Ensure project repo is cloned correctly."
fi

# Step 6: Create .env file with venv activation
cat << EOF > .env
# Activate virtual environment
source ${PROJECT_ROOT}/mlir-venv/bin/activate

# Add MLIR binaries to PATH
export PATH=${PROJECT_ROOT}/llvm-project/build/bin:$PATH

# Add MLIR Python bindings to PYTHONPATH
export PYTHONPATH=${PROJECT_ROOT}/llvm-project/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH


export NEPTUNE_PROJECT="<NEPTUNE_PROJECT_URL>"

export NEPTUNE_TOKEN="<NEPTUNE_API_TOKEN>"

export LLVM_BUILD_PATH=${PROJECT_ROOT}/llvm-project/build

export MLIR_SHARED_LIBS=${PROJECT_ROOT}/llvm-project/build/lib/libomp.so,/home/ouail/nyuad-internship/llvm-project/build/lib/libmlir_c_runner_utils.so,/home/ouail/nyuad-internship/llvm-project/build/lib/libmlir_runner_utils.so

export AST_DUMPER_BIN_PATH=${PROJECT_ROOT}/tools/ast_dumper/build/bin/AstDumper

export VECTORIZER_BIN_PATH=${PROJECT_ROOT}/tools/vectorizer/build/bin/Vectorizer

EOF

echo "Setup complete. Source the env with: source .env (this activates the venv too)."
echo "Edit .env for Neptune if needed."
echo "If tools were skipped, clone the project repo and re-run."
deactivate  # Deactivate venv after script (source .env will reactivate)