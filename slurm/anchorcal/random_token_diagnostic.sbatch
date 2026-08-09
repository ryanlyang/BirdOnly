#!/usr/bin/env bash
#SBATCH --job-name=ac_rt_diag
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/random_token_diagnostics/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/random_token_diagnostics/run_logs/%x_%j.err

set -Eeuo pipefail
: "${ANCHORCAL_DIAGNOSTIC_COMMIT:?submit with submit_random_token_diagnostic.sh}"
: "${ANCHORCAL_PYTHON:?submit with submit_random_token_diagnostic.sh}"

REPO=/home/ryreu/guided_cnn/BirdOnly
actual_commit=$(git -C "$REPO" rev-parse HEAD)
[[ "$actual_commit" == "$ANCHORCAL_DIAGNOSTIC_COMMIT" ]] || {
  echo "diagnostic commit changed: expected=$ANCHORCAL_DIAGNOSTIC_COMMIT actual=$actual_commit" >&2
  exit 2
}
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=normal)" ]] || {
  echo "diagnostic requires a clean checkout" >&2
  exit 2
}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
export PYTHONHASHSEED=8003
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HDF5_USE_FILE_LOCKING=FALSE
export HF_HOME=/home/ryreu/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

target="$REPO/outputs/anchorcal/random_token_diagnostics/reports/random_token_job${SLURM_JOB_ID}.json"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun --unbuffered "$ANCHORCAL_PYTHON" -u \
  scripts/anchorcal/diagnose_random_tokens.py \
  --config configs/anchorcal/pilot.yaml \
  --paths configs/anchorcal/paths.local.yaml \
  --output "$target" \
  --repeats 10
echo "diagnostic_report=$target"
