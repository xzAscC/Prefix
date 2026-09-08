#!/usr/bin/env bash
# A100 jobs are checkpoint-resumable; resubmission continues saved progress.
# Per PostDyn reference: use uv and keep Hugging Face data in the workspace.
#SBATCH -A PAS2324
#SBATCH -q ascend-default
#SBATCH --gres=gpu:a100:1
#SBATCH -t 8:00:00
#SBATCH --requeue
#SBATCH -J prefix-exp${EXP}
#SBATCH -o logs/osc_a100_exp%j.log
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

EXP="${EXP:?EXP must be supplied with --export=EXP=N}"
WORKSPACE="${WORKSPACE:-/fs/ess/PAS2324/zhu.3944/prefix}"
cd "$WORKSPACE"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$WORKSPACE/hf_cache"
mkdir -p logs checkpoints figs

exec uv run python "scripts/run_exp${EXP}.py" \
    --config "configs/exp${EXP}.yaml" \
    --phase direction generate
