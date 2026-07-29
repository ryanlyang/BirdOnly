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
: "${SETV_EXPECTED_COMMIT:?missing SETV_EXPECTED_COMMIT}"
: "${SETV_ULA_REPO:?missing SETV_ULA_REPO}"
: "${SETV_ULA_ENV:?missing SETV_ULA_ENV}"
: "${SETV_ULA_SEED:?missing SETV_ULA_SEED}"
: "${SETV_ULA_SHADOW_DIR:?missing SETV_ULA_SHADOW_DIR}"
: "${SETV_ULA_SSL_DIR:?missing SETV_ULA_SSL_DIR}"

ROOT=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95
MAIN_ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PATH="${SETV_ULA_ENV}/bin:${PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "$SETV_REPO"
actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != "$SETV_EXPECTED_COMMIT" ]] || \
   [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Official uLA SSL refuses changed source after its smoke gate" >&2
  exit 2
fi
"${MAIN_ENV}/bin/python" scripts/audit_ula_source.py \
  --official-repo "$SETV_ULA_REPO"
"${MAIN_ENV}/bin/python" scripts/prepare_ula_official_shadow.py \
  --phase0-dir "${ROOT}/phase0" \
  --output-dir "$SETV_ULA_SHADOW_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
PATH="${SETV_ULA_ENV}/bin:${PATH}" \
  bash scripts/run_official_ula_ssl.sh
test -f "${SETV_ULA_SSL_DIR}/last.ckpt"
"${MAIN_ENV}/bin/python" scripts/smoke_ula_proxy.py \
  --config configs/ula_proxy_waterbirds95.yaml \
  --seed "$SETV_ULA_SEED" \
  --official-repo "$SETV_ULA_REPO" \
  --ssl-checkpoint "${SETV_ULA_SSL_DIR}/last.ckpt" \
  --device cuda \
  --report "${ROOT}/preflight/ula_production_checkpoint_smoke_job${SLURM_JOB_ID}.json"
