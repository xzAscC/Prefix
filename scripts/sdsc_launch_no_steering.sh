#!/usr/bin/env bash

set -euo pipefail

UV_BIN="$(command -v uv)"
WORKSPACE="${1:-/expanse/lustre/projects/ohs122/xzhu11/Prefix}"
HF_HOME="${HF_HOME:-/expanse/lustre/projects/ohs122/xzhu11/hf-cache}"
ASSET_MARKER="${ASSET_MARKER:-$WORKSPACE/.no_steering_assets.ready}"
ASSET_MANIFEST="${ASSET_MANIFEST:-$WORKSPACE/.no_steering_assets.json}"
PREFLIGHT_MARKER="${PREFLIGHT_MARKER:-$WORKSPACE/.no_steering_preflight.ok}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"

die() { printf 'sdsc launch: %s\n' "$1" >&2; exit 1; }
[[ "$UV_BIN" == /* && -x "$UV_BIN" ]] || die "uv is not an executable absolute path: $UV_BIN"
[[ -d "$WORKSPACE" ]] || die "workspace does not exist: $WORKSPACE"
mkdir -p "$WORKSPACE/logs" "$WORKSPACE/checkpoints" "$WORKSPACE/results"
cd "$WORKSPACE"

rm -f "$PREFLIGHT_MARKER"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_ROOT="$WORKSPACE/results/no_steering/runs/$RUN_ID"
PREFLIGHT_MARKER="$RUN_ROOT/preflight.ok"
PREFLIGHT_OUTPUT_ROOT="$RUN_ROOT/preflight"
PREFLIGHT_CHECKPOINT_ROOT="$RUN_ROOT/preflight-checkpoints"
FORMAL_OUTPUT_ROOT="$RUN_ROOT/results"
FORMAL_CHECKPOINT_ROOT="$RUN_ROOT/checkpoints"
mkdir -p "$RUN_ROOT" "$PREFLIGHT_OUTPUT_ROOT" "$PREFLIGHT_CHECKPOINT_ROOT" \
    "$FORMAL_OUTPUT_ROOT" "$FORMAL_CHECKPOINT_ROOT" "$WORKSPACE/logs"
printf 'run_root=%s\n' "$RUN_ROOT"

export HF_HOME HF_DATASETS_CACHE="$HF_HOME" PREFIX_DATA_CACHE="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$UV_BIN" run python "$SCRIPT_DIR/no_steering_preflight.py" \
    --phase check --cache-root "$HF_HOME" \
    --dataset-cache-root "$HF_HOME" \
    --manifest "$ASSET_MANIFEST" --marker "$ASSET_MARKER"

declare -a model_ids=(
    'Qwen/Qwen3-4B'
    'Qwen/Qwen3-14B'
    'allenai/Olmo-3-7B-Think'
    'allenai/Olmo-3-32B-Think'
)
declare -a model_slugs=(
    'Qwen--Qwen3-4B'
    'Qwen--Qwen3-14B'
    'allenai--Olmo-3-7B-Think'
    'allenai--Olmo-3-32B-Think'
)

preflight_job_id=$(sbatch --parsable \
    --export="WORKSPACE=$WORKSPACE,HF_HOME=$HF_HOME,UV_BIN=$UV_BIN,ASSET_MARKER=$ASSET_MARKER,ASSET_MANIFEST=$ASSET_MANIFEST,PREFLIGHT_MARKER=$PREFLIGHT_MARKER,PREFLIGHT_OUTPUT_ROOT=$PREFLIGHT_OUTPUT_ROOT,PREFLIGHT_CHECKPOINT_ROOT=$PREFLIGHT_CHECKPOINT_ROOT,RUN_ROOT=$RUN_ROOT" \
    "$SCRIPT_DIR/sdsc_no_steering_preflight.sbatch") || die "preflight submission failed"
printf 'preflight status=submitted job_id=%s\n' "$preflight_job_id"

status=0
for index in "${!model_ids[@]}"; do
    model_id="${model_ids[$index]}"
    model_slug="${model_slugs[$index]}"
    if job_id=$(sbatch --parsable "--dependency=afterok:${preflight_job_id}" \
        --export="WORKSPACE=$WORKSPACE,HF_HOME=$HF_HOME,UV_BIN=$UV_BIN,MODEL_ID=$model_id,MODEL_SLUG=$model_slug,ASSET_MARKER=$ASSET_MARKER,ASSET_MANIFEST=$ASSET_MANIFEST,PREFLIGHT_MARKER=$PREFLIGHT_MARKER,RUN_ROOT=$RUN_ROOT,FORMAL_OUTPUT_ROOT=$FORMAL_OUTPUT_ROOT,CHECKPOINT_DIR=$FORMAL_CHECKPOINT_ROOT" \
        "$SCRIPT_DIR/sdsc_no_steering_job.sbatch"); then
        printf 'model=%s status=submitted job_id=%s\n' "$model_id" "$job_id"
    else
        printf 'model=%s status=submit-failed\n' "$model_id" >&2
        status=1
    fi
done
exit "$status"
