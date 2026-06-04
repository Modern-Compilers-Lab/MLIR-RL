#!/bin/bash
# Hands-free entry point for the LLM Transform framework.
#
# Ensures the framework is fully set up (running scripts/setup.sh if not),
# checks that everything needed is in place (conda env, equivalence verifier
# plugins, the `claude` CLI), then launches a Claude optimization session via
# scripts/claude.sh, waits for it to finish, and reports where the results were
# saved.
#
# Usage:
#   bash   scripts/run.sh [benchmark|id ...]   # submits claude.sh to Slurm (sbatch) and waits
#   sbatch scripts/run.sh [benchmark|id ...]   # runs claude.sh inline in this allocation
#
# With no benchmark/id arguments you are prompted for an optional filter
# (a non-interactive run with no arguments optimizes every instance in data/).

#SBATCH -J llm_transform
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 7-00
#SBATCH -o logs/claude/%j.log

set -eo pipefail

# --- locate the project root (parent of this script's directory) -------------
# resolve_root <path> echoes the project root for a run.sh located at <path>,
# or fails if <path> is empty / doesn't sit at <root>/scripts/run.sh.
resolve_root() {
    local self="$1" root
    [ -n "$self" ] || return 1
    root="$(cd "$(dirname "$(dirname "$(realpath "$self")")")" 2>/dev/null && pwd)" || return 1
    [ -f "$root/scripts/run.sh" ] || return 1
    printf '%s\n' "$root"
}

# Was THIS job submitted as `sbatch run.sh`? Only then is the live BASH_SOURCE a
# throwaway spool copy and the job's recorded Command the real run.sh path. If
# run.sh is merely running inside some other allocation (an interactive session,
# or an sbatch of a different script) the Command is NOT run.sh, while
# BASH_SOURCE is the genuine path -- so the job Command is the deciding signal.
ROOT=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SELF=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | cut -d' ' -f1)
    if ROOT="$(resolve_root "$SELF")"; then
        echo ">> Submitted via sbatch (job $SLURM_JOB_ID); project root from job Command."
    else
        ROOT=""
    fi
fi
if [ -z "$ROOT" ]; then
    ROOT="$(resolve_root "${BASH_SOURCE[0]}")" || true
fi
if [ -z "$ROOT" ]; then
    echo "Error: could not locate the project root (no run.sh found from the Slurm" >&2
    echo "       job Command or this script's path)." >&2
    exit 1
fi
cd "$ROOT"
echo ">> Project root: $ROOT"

# --- bring conda into the shell ---------------------------------------------
module load miniconda-nobashrc 2>/dev/null || true
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: 'conda' was not found. Install Miniconda/Anaconda (or load your" >&2
    echo "       cluster's conda module) and re-run this script." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

# --- is the framework fully set up? -----------------------------------------
setup_incomplete() {
    [ -f "$ROOT/scripts/env.local.sh" ] || return 0
    # shellcheck disable=SC1091
    source "$ROOT/scripts/env.local.sh"
    [ -n "${MAIN_ENV:-}" ] || return 0
    local eqlib="$ROOT/llm_transform/tools/c/equivalence/build/lib"
    for so in libEquivalenceVerifier.so libTagLinalgOps.so libRaiseSCFToAffine.so; do
        [ -f "$eqlib/$so" ] || return 0
    done
    conda env list | awk '{print $1}' | grep -Fxq "$MAIN_ENV" || return 0
    return 1
}

if setup_incomplete; then
    echo ">> Framework not set up yet; running scripts/setup.sh ..."
    bash "$ROOT/scripts/setup.sh"
fi

# Load MAIN_ENV (now guaranteed to exist).
# shellcheck disable=SC1091
source "$ROOT/scripts/env.local.sh"
echo ">> Using conda environment: $MAIN_ENV"

# --- check the claude CLI is installed --------------------------------------
if ! command -v claude >/dev/null 2>&1; then
    echo "Error: the 'claude' CLI was not found on PATH." >&2
    echo "       Install Claude Code and authenticate it before running:" >&2
    echo "         https://docs.claude.com/en/docs/claude-code" >&2
    echo "         claude login" >&2
    exit 1
fi

# --- choose the benchmark/instance filter -----------------------------------
if [ "$#" -ge 1 ]; then
    FILTER=("$@")
elif [ -t 0 ]; then
    read -rp "Filter benchmarks/instances (space-separated, empty = all): " -a FILTER
else
    FILTER=()
fi
echo ">> Instance filter: ${FILTER[*]:-ALL}"

# --- run the Claude session and capture the experiment id -------------------
EXP_ID=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Already inside a Slurm allocation: run claude.sh as a normal script.
    echo ">> Running scripts/claude.sh inline (Slurm allocation $SLURM_JOB_ID) ..."
    RUN_LOG=$(mktemp)
    set +e
    bash "$ROOT/scripts/claude.sh" "${FILTER[@]}" 2>&1 | tee "$RUN_LOG"
    status=${PIPESTATUS[0]}
    set -e
    EXP_ID=$(grep -oE 'Experiment ID: *[0-9]+' "$RUN_LOG" | grep -oE '[0-9]+' | tail -1)
    rm -f "$RUN_LOG"
else
    # Login node: submit claude.sh to Slurm and block until it finishes.
    echo ">> Submitting scripts/claude.sh to Slurm and waiting for it to finish ..."
    set +e
    JOB_ID=$(sbatch --parsable --wait "$ROOT/scripts/claude.sh" "${FILTER[@]}")
    status=$?
    set -e
    JOB_ID=${JOB_ID%%;*}
    echo ">> Slurm job $JOB_ID finished."
    JOB_LOG="$ROOT/logs/claude/$JOB_ID.log"
    if [ -f "$JOB_LOG" ]; then
        EXP_ID=$(grep -oE 'Experiment ID: *[0-9]+' "$JOB_LOG" | grep -oE '[0-9]+' | tail -1)
    fi
fi

# --- report ------------------------------------------------------------------
echo
if [ "${status:-0}" -ne 0 ]; then
    echo "The Claude session exited with a non-zero status ($status)." >&2
fi
if [ -z "$EXP_ID" ] || [ ! -d "$ROOT/logs/stats/$EXP_ID" ]; then
    echo "Error: could not determine the experiment directory from the Claude session" >&2
    echo "       output; check logs/stats/ and logs/claude/." >&2
    exit 1
fi

echo "Results saved to: logs/stats/$EXP_ID/"
echo "  - claude_optimization.log   every run_schedule attempt"
echo "  - performance.png           speedup-over-time plot"
echo "  - tokens.log                input/output token counts per turn"
echo "  - best/                     best configuration per instance"

exit "${status:-0}"
