#!/usr/bin/env bash

set -euo pipefail

RESPONSE_ROOT="${1:?usage: $0 RESPONSE_ROOT OUTPUT_ROOT [JUDGE_MODEL]}"
OUTPUT_ROOT="${2:?usage: $0 RESPONSE_ROOT OUTPUT_ROOT [JUDGE_MODEL]}"
JUDGE_MODEL="${3:-gemini-3.5-flash-lite}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
if [[ "$RESPONSE_ROOT" == */checkpoints ]]; then
    GENERATION_ROOT="${RESPONSE_ROOT%/checkpoints}"
else
    printf 'response root must end in /checkpoints: %s\n' "$RESPONSE_ROOT" >&2
    exit 2
fi
STATE_PATH="$OUTPUT_ROOT/finalization.json"

models=(
    'Qwen/Qwen3-4B'
    'Qwen/Qwen3-14B'
    'allenai/Olmo-3-7B-Think'
    'allenai/Olmo-3-32B-Think'
)

for model_id in "${models[@]}"; do
    uv run python "$SCRIPT_DIR/run_no_steering.py" \
        --model-id "$model_id" \
        --phase score \
        --checkpoint-root "$RESPONSE_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --judge-model "$JUDGE_MODEL"
done

uv run python "$SCRIPT_DIR/finalize_no_steering.py" \
    --generation-root "$GENERATION_ROOT" \
    --scoring-root "$OUTPUT_ROOT" \
    --state-path "$STATE_PATH"
