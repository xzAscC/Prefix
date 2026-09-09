#!/usr/bin/env bash
# Load and generate one bounded sample for every staged model after validation.
#SBATCH --account=ohs122
#SBATCH --partition=nairr-gpu-shared
#SBATCH --qos=nairr-gpu-shared-normal
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/sdsc_no_steering_preflight_%j.log

set -euo pipefail

: "${UV_BIN:?UV_BIN must be supplied explicitly}"
[[ "$UV_BIN" == /* && -x "$UV_BIN" ]] || {
    printf 'UV_BIN is not an executable absolute path: %s\n' "$UV_BIN" >&2
    exit 1
}
WORKSPACE="${WORKSPACE:-/expanse/lustre/projects/ohs122/xzhu11/Prefix}"
HF_HOME="${HF_HOME:-/expanse/lustre/projects/ohs122/xzhu11/hf-cache}"
ASSET_MARKER="${ASSET_MARKER:-$WORKSPACE/.no_steering_assets.ready}"
ASSET_MANIFEST="${ASSET_MANIFEST:-$WORKSPACE/.no_steering_assets.json}"
PREFLIGHT_MARKER="${PREFLIGHT_MARKER:-$WORKSPACE/.no_steering_preflight.ok}"
PREFLIGHT_OUTPUT_ROOT="${PREFLIGHT_OUTPUT_ROOT:-$WORKSPACE/results/no_steering/preflight}"
PREFLIGHT_CHECKPOINT_ROOT="${PREFLIGHT_CHECKPOINT_ROOT:-$WORKSPACE/checkpoints/no_steering/preflight}"

cd "$WORKSPACE"
mkdir -p logs checkpoints "$PREFLIGHT_OUTPUT_ROOT" "$PREFLIGHT_CHECKPOINT_ROOT"
rm -f "$PREFLIGHT_MARKER"

export HF_HOME HF_DATASETS_CACHE="$HF_HOME" PREFIX_DATA_CACHE="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$UV_BIN" run python scripts/no_steering_preflight.py \
    --phase check --cache-root "$HF_HOME" --dataset-cache-root "$HF_HOME"

model_ids=(
    'Qwen/Qwen3-4B'
    'Qwen/Qwen3-14B'
    'allenai/Olmo-3-7B-Think'
    'allenai/Olmo-3-32B-Think'
)

for model_id in "${model_ids[@]}"; do
    model_slug="${model_id//\//--}"
    "$UV_BIN" run python scripts/run_no_steering.py \
        --model-id "$model_id" \
        --phase generate \
        --limit 1 \
        --max-tokens 8 \
        --max-model-len 512 \
        --gpu-memory-utilization 0.85 \
        --cache-root "$HF_HOME" \
        --checkpoint-root "$PREFLIGHT_CHECKPOINT_ROOT" \
        --output-root "$PREFLIGHT_OUTPUT_ROOT"
    summary="$PREFLIGHT_OUTPUT_ROOT/$model_slug/summary.json"
    [[ -s "$summary" ]] || {
        printf 'missing smoke output for %s: %s\n' "$model_id" "$summary" >&2
        exit 1
    }
    "$UV_BIN" run python scripts/validate_no_steering_smoke.py \
        --result-root "$PREFLIGHT_OUTPUT_ROOT/$model_slug" \
        --checkpoint-root "$PREFLIGHT_CHECKPOINT_ROOT/$model_slug" \
        --model-id "$model_id"
done

tmp_marker="$(mktemp "${PREFLIGHT_MARKER}.tmp.XXXXXX")"
cleanup_marker() {
    rm -f -- "$tmp_marker"
}
trap cleanup_marker EXIT
{
    printf 'preflight_job=%s\n' "${SLURM_JOB_ID:-local}"
    printf 'output_root=%s\n' "$PREFLIGHT_OUTPUT_ROOT"
    printf 'models=%s\n' "${model_ids[*]}"
} > "$tmp_marker"
mv -f "$tmp_marker" "$PREFLIGHT_MARKER"
