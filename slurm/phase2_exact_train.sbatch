#!/usr/bin/env bash
#SBATCH --job-name=setv_exact_train
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.err

set -Eeuo pipefail
: "${SETV_REPO:?missing SETV_REPO}"
: "${SETV_EXACT_SEED:?missing SETV_EXACT_SEED}"

ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
OUTPUT="/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/background_exact/seed_${SETV_EXACT_SEED}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$SETV_REPO"
echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun --unbuffered python -u scripts/train_exact_expert.py \
  --config configs/expert_background_exact.yaml \
  --seed "$SETV_EXACT_SEED"
python -u scripts/verify_exact_expert.py --output-dir "$OUTPUT"

