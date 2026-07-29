#!/usr/bin/env bash
#SBATCH --job-name=setv_phase6
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.err

set -Eeuo pipefail
: "${SETV_REPO:?missing SETV_REPO}"
: "${SETV_CANDIDATE_SEEDS:?missing SETV_CANDIDATE_SEEDS}"
: "${SETV_ULA_PROXY_DIR:?missing SETV_ULA_PROXY_DIR}"
: "${SETV_EXACT_FUSION_DIR:?missing SETV_EXACT_FUSION_DIR}"
: "${SETV_SANITIZED_FUSION_DIR:?missing SETV_SANITIZED_FUSION_DIR}"
: "${SETV_SET_FUSION_DIR:?missing SETV_SET_FUSION_DIR}"

ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
ROOT=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$SETV_REPO"
python -u scripts/run_phase6_analysis.py \
  --config configs/phase6_analysis.yaml \
  --candidate-seeds "$SETV_CANDIDATE_SEEDS" \
  --ula-proxy-dir "$SETV_ULA_PROXY_DIR" \
  --exact-fusion-dir "$SETV_EXACT_FUSION_DIR" \
  --sanitized-fusion-dir "$SETV_SANITIZED_FUSION_DIR" \
  --set-fusion-dir "$SETV_SET_FUSION_DIR"
python -u scripts/verify_phase6.py --output-dir "${ROOT}/phase6"
