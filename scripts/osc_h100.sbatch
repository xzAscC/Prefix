#!/usr/bin/env bash
# H100 jobs are preemptible 4-hour blocks. The entrypoint is fully
# checkpoint-resumable, so resubmitted blocks continue from saved progress.
# Per PostDyn reference: use uv and keep Hugging Face data in the workspace.
#SBATCH -A PAS2324
#SBATCH -q cardinal-default
#SBATCH --gres=gpu:h100:1
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH -o logs/osc_h100_exp%j.log
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

EXP="${EXP:?EXP must be supplied with --export=EXP=N}"
WORKSPACE="${WORKSPACE:-/fs/ess/PAS2324/zhu.3944/prefix}"
cd "$WORKSPACE"

export PATH="$HOME/.local/bin:/fs/ess/PAS2324/zhu.3944/google-cloud-sdk/bin:$PATH"
export HF_HOME="${HF_HOME:-/fs/ess/PAS2324/zhu.3944/hf-cache}"
mkdir -p logs checkpoints figs

exec uv run python "scripts/run_exp${EXP}.py" \
    --config "configs/exp${EXP}.yaml" \
    --phase direction generate
