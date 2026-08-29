#!/bin/bash
# Create the unversioned MLIR runner-utils dev symlinks that ld needs at link time.
#
# The conda-forge `libmlir21` package ships only the versioned shared objects
# (libmlir_runner_utils.so.<ver>, libmlir_c_runner_utils.so.<ver>) and omits the
# unversioned `.so` symlinks. Without them, `gcc ... -lmlir_runner_utils
# -lmlir_c_runner_utils` (see llm_transform/utils/transformation.py) fails with
# "cannot find -lmlir_runner_utils". This script recreates those symlinks.
#
# Idempotent and version-agnostic: re-running is a no-op, and it links whatever
# version conda installed. Run it once after creating/activating $MAIN_ENV.
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Error: CONDA_PREFIX is not set. Activate the conda environment first:" >&2
    echo "  source scripts/env.local.sh && conda activate \"\$MAIN_ENV\"" >&2
    exit 1
fi

lib_dir="$CONDA_PREFIX/lib"
for base in libMLIR libmlir_runner_utils libmlir_c_runner_utils; do
    if [ -e "$lib_dir/$base.so" ]; then
        echo "ok: $base.so already present"
        continue
    fi
    # Pick the highest installed version, e.g. libmlir_runner_utils.so.21.1
    versioned=$(ls -1 "$lib_dir/$base".so.* 2>/dev/null | sort -V | tail -n1 || true)
    if [ -z "$versioned" ]; then
        echo "Error: no $base.so.* found in $lib_dir; is 'mlir' installed in this env?" >&2
        exit 1
    fi
    ln -s "$(basename "$versioned")" "$lib_dir/$base.so"
    echo "linked $base.so -> $(basename "$versioned")"
done
