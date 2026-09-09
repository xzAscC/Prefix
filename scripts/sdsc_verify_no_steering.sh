#!/usr/bin/env bash

set -uo pipefail

WORKSPACE="${1:-/expanse/lustre/projects/ohs122/xzhu11/Prefix}"
HF_HOME="${HF_HOME:-/expanse/lustre/projects/ohs122/xzhu11/hf-cache}"
ASSET_MARKER="${ASSET_MARKER:-$WORKSPACE/.no_steering_assets.ready}"
ASSET_MANIFEST="${ASSET_MANIFEST:-$WORKSPACE/.no_steering_assets.json}"
RUN_ROOT="${RUN_ROOT:-${2:-}}"
if [[ -z "$RUN_ROOT" ]]; then
    printf 'usage: %s WORKSPACE RUN_ROOT\n' "$0" >&2
    exit 2
fi
PREFLIGHT_MARKER="${PREFLIGHT_MARKER:-$RUN_ROOT/preflight.ok}"
PREFLIGHT_OUTPUT_ROOT="${PREFLIGHT_OUTPUT_ROOT:-$RUN_ROOT/preflight}"

errors=0
cd "$WORKSPACE"
check_file() {
    if [[ ! -f "$1" ]]; then
        printf 'missing: %s\n' "$1" >&2
        errors=1
    else
        printf 'present: %s\n' "$1"
    fi
}
check_file "$PREFLIGHT_MARKER"

check_smoke_output() {
    local model_id="$1"
    local model_slug="$2"
    local output="$PREFLIGHT_OUTPUT_ROOT/$model_slug/summary.json"
    check_file "$output"
}

check_smoke_output 'Qwen/Qwen3-4B' 'Qwen--Qwen3-4B'
check_smoke_output 'Qwen/Qwen3-14B' 'Qwen--Qwen3-14B'
check_smoke_output 'allenai/Olmo-3-7B-Think' 'allenai--Olmo-3-7B-Think'
check_smoke_output 'allenai/Olmo-3-32B-Think' 'allenai--Olmo-3-32B-Think'

for model_slug in Qwen--Qwen3-4B Qwen--Qwen3-14B allenai--Olmo-3-7B-Think allenai--Olmo-3-32B-Think; do
    check_file "$RUN_ROOT/results/$model_slug/summary.json"
    check_file "$RUN_ROOT/checkpoints/$model_slug/manifest.json"
    for benchmark in harmbench mmlu_pro math500; do
        check_file "$RUN_ROOT/checkpoints/$model_slug/$benchmark/responses.jsonl"
    done
    model_id="${model_slug//--//}"
    if uv run python scripts/validate_no_steering_formal.py \
        --result-root "$RUN_ROOT/results/$model_slug" \
        --checkpoint-root "$RUN_ROOT/checkpoints/$model_slug" \
        --model-id "$model_id"; then
        printf 'formal validation: OK (%s)\n' "$model_id"
    else
        printf 'formal validation: FAILED (%s)\n' "$model_id" >&2
        errors=1
    fi
done

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
if uv run python scripts/no_steering_preflight.py \
    --phase check --cache-root "$HF_HOME" \
    --manifest "$ASSET_MANIFEST" --marker "$ASSET_MARKER"; then
    printf 'asset validation: OK\n'
else
    printf 'asset validation: FAILED\n' >&2
    errors=1
fi

if (( errors )); then
    printf 'verification: FAILED\n' >&2
    exit 1
fi
printf 'verification: OK\n'
