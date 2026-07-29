#!/usr/bin/env bash
#SBATCH --job-name=setv_sanitized_audit
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.err

set -Eeuo pipefail
: "${SETV_REPO:?missing SETV_REPO}"
: "${SETV_MASK_SEED:?missing SETV_MASK_SEED}"

ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
BANK="/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/sanitized_mask_bank/seed_${SETV_MASK_SEED}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$SETV_REPO"
echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse HEAD)"
python -c "import torch,scipy,sklearn; print(torch.__version__,scipy.__version__,sklearn.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun --unbuffered python -u scripts/build_sanitized_mask_bank.py \
  --config configs/sanitized_mask_bank.yaml \
  --seed "$SETV_MASK_SEED"
python -u scripts/verify_sanitized_mask_bank.py --mask-bank-dir "$BANK"

