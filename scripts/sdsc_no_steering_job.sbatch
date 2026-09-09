#!/usr/bin/env bash
# One model per job; all state is written to durable checkpoint/output paths.
#SBATCH --account=ohs122
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/sdsc_no_steering_%j.log

set -euo pipefail

: "${UV_BIN:?UV_BIN must be supplied explicitly}"
: "${MODEL_ID:?MODEL_ID must be supplied explicitly}"
: "${MODEL_SLUG:?MODEL_SLUG must be supplied explicitly}"
: "${RUN_ROOT:?RUN_ROOT must be supplied explicitly}"
[[ "$UV_BIN" == /* && -x "$UV_BIN" ]] || {
    printf 'UV_BIN is not an executable absolute path: %s\n' "$UV_BIN" >&2
    exit 1
}
[[ "$MODEL_SLUG" == "${MODEL_ID//\//--}" ]] || {
    printf 'model slug does not match canonical model path: %s\n' "$MODEL_SLUG" >&2
    exit 1
}
WORKSPACE="${WORKSPACE:-/expanse/lustre/projects/ohs122/xzhu11/Prefix}"
HF_HOME="${HF_HOME:-/expanse/lustre/projects/ohs122/xzhu11/hf-cache}"
ASSET_MARKER="${ASSET_MARKER:-$WORKSPACE/.no_steering_assets.ready}"
ASSET_MANIFEST="${ASSET_MANIFEST:-$WORKSPACE/.no_steering_assets.json}"
PREFLIGHT_MARKER="${PREFLIGHT_MARKER:-$WORKSPACE/.no_steering_preflight.ok}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-$RUN_ROOT/results}"

cd "$WORKSPACE"
mkdir -p logs "$CHECKPOINT_DIR" "$FORMAL_OUTPUT_ROOT"
[[ -f "$PREFLIGHT_MARKER" ]] || { printf 'missing preflight marker: %s\n' "$PREFLIGHT_MARKER" >&2; exit 1; }
export HF_HOME HF_DATASETS_CACHE="$HF_HOME" PREFIX_DATA_CACHE="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$UV_BIN" run python scripts/no_steering_preflight.py \
    --phase check --model-id "$MODEL_ID" --cache-root "$HF_HOME" \
    --dataset-cache-root "$HF_HOME"
"$UV_BIN" run python scripts/run_no_steering.py \
    --model-id "$MODEL_ID" \
    --phase generate \
    --cache-root "$HF_HOME" \
    --checkpoint-root "$CHECKPOINT_DIR" \
    --output-root "$FORMAL_OUTPUT_ROOT"
