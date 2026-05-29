#!/usr/bin/env bash
# PreToolUse hook for Read/Glob/Grep:
# - Denies access to logs/stats/ unless the path is inside $EXPERIMENT_DIR
#   (set by scripts/claude.sh for the current experiment).

set -euo pipefail

input=$(cat)
tool=$(printf '%s' "$input" | jq -r '.tool_name // ""')

case "$tool" in
    Read)  raw=$(printf '%s' "$input" | jq -r '.tool_input.file_path // ""') ;;
    Glob)  raw=$(printf '%s' "$input" | jq -r '.tool_input.path // ""') ;;
    Grep)  raw=$(printf '%s' "$input" | jq -r '.tool_input.path // ""') ;;
    *)     exit 0 ;;
esac

[[ -z "$raw" ]] && exit 0

project_root=$(realpath -- "${CLAUDE_PROJECT_DIR:-$(dirname -- "$(dirname -- "$(dirname -- "${BASH_SOURCE[0]}")")")}")
abs=$(cd "$project_root" 2>/dev/null && realpath -m -- "$raw" 2>/dev/null || printf '%s' "$raw")

stats_root="$project_root/logs/stats"

deny() {
    jq -nc --arg reason "$1" '{
        hookSpecificOutput: {
            hookEventName: "PreToolUse",
            permissionDecision: "deny",
            permissionDecisionReason: $reason
        }
    }'
    exit 0
}

if [[ "$abs" == "$stats_root" || "$abs" == "$stats_root"/* ]]; then
    if [[ -n "${EXPERIMENT_DIR:-}" ]]; then
        exp_abs=$(cd "$project_root" 2>/dev/null && realpath -m -- "$EXPERIMENT_DIR" 2>/dev/null || printf '%s' "$EXPERIMENT_DIR")
        if [[ "$abs" == "$exp_abs" || "$abs" == "$exp_abs"/* ]]; then
            exit 0
        fi
    fi
    deny "Access to logs/stats/ is restricted (only the current EXPERIMENT_DIR is permitted)."
fi

exit 0
