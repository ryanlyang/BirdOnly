#!/usr/bin/env bash
#SBATCH --job-name=setv_ula_ssl
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
: "${SETV_ULA_REPO:?missing SETV_ULA_REPO}"
: "${SETV_ULA_ENV:?missing SETV_ULA_ENV}"
: "${SETV_ULA_SHADOW_DIR:?missing SETV_ULA_SHADOW_DIR}"
: "${SETV_ULA_SSL_DIR:?missing SETV_ULA_SSL_DIR}"

ROOT=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PATH="${SETV_ULA_ENV}/bin:${PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "$SETV_REPO"
"${SETV_ULA_ENV}/bin/python" scripts/audit_ula_source.py \
  --official-repo "$SETV_ULA_REPO"
"${SETV_ULA_ENV}/bin/python" scripts/prepare_ula_official_shadow.py \
  --phase0-dir "${ROOT}/phase0" \
  --output-dir "$SETV_ULA_SHADOW_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
PATH="${SETV_ULA_ENV}/bin:${PATH}" \
  bash scripts/run_official_ula_ssl.sh
test -f "${SETV_ULA_SSL_DIR}/last.ckpt"
