#!/bin/bash
# One-shot setup for the LLM Transform framework.
#
# Creates (or reuses) a conda environment with the full MLIR/LLVM build
# toolchain, installs the Python package and its dependencies with poetry,
# writes the per-user scripts/env.local.sh, and performs every build the
# framework needs (MLIR runner-utils dev symlinks + the equivalence verifier
# plugins). After this finishes you can run the framework directly.
#
# Usage:
#   bash scripts/setup.sh                 # prompts for the conda env name
#   bash scripts/setup.sh <env-name>      # uses <env-name> without prompting
#   MAIN_ENV=<env-name> bash scripts/setup.sh
#
# Re-running is safe: it is idempotent and updates an existing environment.
set -eo pipefail

DEFAULT_ENV="llm_transform"

# --- locate the project root (parent of this script's directory) -------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- bring conda into the shell ---------------------------------------------
module load miniconda-nobashrc 2>/dev/null || true
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: 'conda' was not found. Install Miniconda/Anaconda (or load your" >&2
    echo "       cluster's conda module) and re-run this script." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

# --- choose the environment name --------------------------------------------
if [ -n "${MAIN_ENV:-}" ]; then
    ENV_NAME="$MAIN_ENV"
elif [ "$#" -ge 1 ]; then
    ENV_NAME="$1"
elif [ -t 0 ]; then
    read -rp "Conda environment name [${DEFAULT_ENV}]: " ENV_NAME
    ENV_NAME="${ENV_NAME:-$DEFAULT_ENV}"
else
    ENV_NAME="$DEFAULT_ENV"
fi
echo ">> Using conda environment: $ENV_NAME"

# --- create the environment if it does not already exist --------------------
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo ">> Environment '$ENV_NAME' already exists; updating its packages."
else
    echo ">> Creating conda environment '$ENV_NAME'."
    conda create -y -n "$ENV_NAME"
fi

# --- install the conda-managed toolchain + dependencies ---------------------
echo ">> Installing conda packages (MLIR/LLVM toolchain, poetry, build tools)."
conda install -y -n "$ENV_NAME" -c conda-forge \
    python=3.11 \
    poetry \
    cmake \
    ninja \
    jq \
    re2c \
    conda-gcc-specs \
    gxx_linux-64 \
    clang=21.1.8 \
    clangxx=21.1.8 \
    lld=21.1.8 \
    llvm-openmp=21.1.8 \
    mlir-python-bindings=21.1.8

# --- activate so the remaining steps use this environment's tools -----------
conda activate "$ENV_NAME"

# --- write the per-user env.local.sh (consumed by the other scripts/MCP) ----
ENV_LOCAL="$ROOT/scripts/env.local.sh"
if [ -f "$ENV_LOCAL" ]; then
    echo ">> Backing up existing scripts/env.local.sh to env.local.sh.bak"
    cp "$ENV_LOCAL" "$ENV_LOCAL.bak"
fi
cat > "$ENV_LOCAL" <<EOF
# Per-user conda environment names (gitignored).
export MAIN_ENV=$ENV_NAME
EOF
echo ">> Wrote scripts/env.local.sh (MAIN_ENV=$ENV_NAME)."

# --- install the Python package + dependencies into the conda env -----------
# virtualenvs.create=false makes poetry install into the active conda env
# instead of spawning its own virtualenv, so `python -m llm_transform.*`
# resolves whenever the env is activated.
echo ">> Installing Python dependencies with poetry."
POETRY_VIRTUALENVS_CREATE=false poetry install

# --- create the unversioned MLIR runner-utils dev symlinks ld needs ---------
echo ">> Creating MLIR runner-utils dev symlinks."
bash "$ROOT/scripts/setup-mlir-links.sh"

# --- build the equivalence verifier plugins ---------------------------------
echo ">> Building the equivalence verifier plugins."
make -C "$ROOT/llm_transform/tools/c/equivalence"

echo
echo "Setup complete. To start using the framework:"
echo "    source scripts/env.local.sh"
echo "    conda activate \"\$MAIN_ENV\""
if ! command -v claude >/dev/null 2>&1; then
    echo
    echo "You will also need Claude Code installed and authenticated (claude login)."
fi
