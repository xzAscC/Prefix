#!/usr/bin/env bash

set -euo pipefail

RESPONSE_ROOT="${1:?usage: $0 RESPONSE_ROOT OUTPUT_ROOT [JUDGE_MODEL]}"
OUTPUT_ROOT="${2:?usage: $0 RESPONSE_ROOT OUTPUT_ROOT [JUDGE_MODEL]}"
JUDGE_MODEL="${3:-gemini-3.5-flash-lite}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
: "${UV_BIN:?UV_BIN must be supplied explicitly}"
if [[ "$UV_BIN" != /* || ! -x "$UV_BIN" ]]; then
    printf 'invalid UV_BIN: %s\n' "$UV_BIN" >&2
    exit 1
fi
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [[ "$RESPONSE_ROOT" == */checkpoints ]]; then
    GENERATION_ROOT="${RESPONSE_ROOT%/checkpoints}"
else
    printf 'response root must end in /checkpoints: %s\n' "$RESPONSE_ROOT" >&2
    exit 2
fi
STATE_PATH="$GENERATION_ROOT/notification.json"

models=(
    'Qwen/Qwen3-4B'
    'Qwen/Qwen3-14B'
    'allenai/Olmo-3-7B-Think'
    'allenai/Olmo-3-32B-Think'
)

child_status=0
for model_id in "${models[@]}"; do
    if "$UV_BIN" run --offline --frozen --no-sync python "$SCRIPT_DIR/run_no_steering.py" \
        --model-id "$model_id" \
        --phase score \
        --managed-finalizer \
        --checkpoint-root "$RESPONSE_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --judge-model "$JUDGE_MODEL"; then
        :
    else
        child_status=$?
        break
    fi
done

finalizer_status=0
if "$UV_BIN" run --offline --frozen --no-sync python "$SCRIPT_DIR/finalize_no_steering.py" \
    --generation-root "$GENERATION_ROOT" \
    --scoring-root "$OUTPUT_ROOT" \
    --state-path "$STATE_PATH"; then
    :
else
    finalizer_status=$?
fi

if (( child_status != 0 )); then
    exit "$child_status"
fi
exit "$finalizer_status"
