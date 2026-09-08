#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf 'Usage: %s a100|h100|cpu [workspace]\n' "$0" >&2
    exit 2
}

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || usage
TYPE="$1"
case "$TYPE" in
    a100|h100|cpu) ;;
    *) usage ;;
esac

WORKSPACE="${2:-/fs/ess/PAS2324/zhu.3944/prefix}"
for EXP in 1 2 3 4; do
    JOB_ID=$(sbatch --parsable \
        --export="ALL,EXP=$EXP,WORKSPACE=$WORKSPACE" \
        "scripts/osc_${TYPE}.sbatch")
    printf 'exp%s: %s\n' "$EXP" "$JOB_ID"
done
