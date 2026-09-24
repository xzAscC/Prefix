#!/usr/bin/env bash
# Supply allocation, partition, GPU, time, and output options at submission.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the prepared project workspace}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

uv run --no-sync python -u scripts/run_section4_native.py extract
uv run --no-sync python -u scripts/run_section4_native.py analyze
