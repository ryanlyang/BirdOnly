#!/usr/bin/env bash
#SBATCH --job-name=setv_sanitized_train
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.err

set -Eeuo pipefail
: "${SETV_REPO:?missing SETV_REPO}"
: "${SETV_MASK_SEED:?missing SETV_MASK_SEED}"
: "${SETV_SANITIZED_SEED:?missing SETV_SANITIZED_SEED}"
if [[ "${SETV_ALLOW_REJECTED_SANITIZED_BANK:-0}" == "1" ]]; then
  override_args=(--allow-rejected-mask-bank)
else
  override_args=()
fi

ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
ROOT=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95
BANK="${ROOT}/sanitized_mask_bank/seed_${SETV_MASK_SEED}"
OUTPUT="${ROOT}/background_sanitized/seed_${SETV_SANITIZED_SEED}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$SETV_REPO"
echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun --unbuffered python -u scripts/train_sanitized_expert.py \
  --config configs/expert_background_sanitized.yaml \
  --seed "$SETV_SANITIZED_SEED" \
  --mask-bank-dir "$BANK" \
  "${override_args[@]}"
python -u scripts/verify_sanitized_expert.py --output-dir "$OUTPUT"
