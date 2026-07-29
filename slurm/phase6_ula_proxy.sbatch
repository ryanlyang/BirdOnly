#!/usr/bin/env bash
#SBATCH --job-name=setv_ula_proxy
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.err

set -Eeuo pipefail
: "${SETV_REPO:?missing SETV_REPO}"
: "${SETV_ULA_REPO:?missing SETV_ULA_REPO}"
: "${SETV_ULA_SEED:?missing SETV_ULA_SEED}"
: "${SETV_ULA_SSL_CHECKPOINT:?missing SETV_ULA_SSL_CHECKPOINT}"

ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
ROOT=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$SETV_REPO"
python -u scripts/train_ula_proxy.py \
  --config configs/ula_proxy_waterbirds95.yaml \
  --seed "$SETV_ULA_SEED" \
  --official-repo "$SETV_ULA_REPO" \
  --ssl-checkpoint "$SETV_ULA_SSL_CHECKPOINT"
python -u scripts/verify_ula_proxy.py \
  --output-dir "${ROOT}/ula_proxy/seed_${SETV_ULA_SEED}"
