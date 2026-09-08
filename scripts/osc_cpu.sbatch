#!/usr/bin/env bash
# Run after GPU phases have produced generations. The judge phase fails
# gracefully on incomplete checkpoints via require_complete.
#SBATCH -A PAS2324
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH -o logs/osc_cpu_exp%j.log

set -euo pipefail

EXP="${EXP:?EXP must be supplied with --export=EXP=N}"
WORKSPACE="${WORKSPACE:-/fs/ess/PAS2324/zhu.3944/prefix}"
cd "$WORKSPACE"

export PATH="$HOME/.local/bin:/fs/ess/PAS2324/zhu.3944/google-cloud-sdk/bin:$PATH"
export HF_HOME="${HF_HOME:-/fs/ess/PAS2324/zhu.3944/hf-cache}"
mkdir -p logs checkpoints figs

exec uv run python "scripts/run_exp${EXP}.py" \
    --config "configs/exp${EXP}.yaml" \
    --phase judge analyze
