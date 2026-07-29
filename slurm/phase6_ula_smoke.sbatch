#!/usr/bin/env bash
#SBATCH --job-name=setv_ula_smoke
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.err

set -Eeuo pipefail
: "${SETV_REPO:?missing SETV_REPO}"
: "${SETV_EXPECTED_COMMIT:?missing SETV_EXPECTED_COMMIT}"
: "${SETV_ULA_REPO:?missing SETV_ULA_REPO}"
: "${SETV_ULA_SEED:?missing SETV_ULA_SEED}"
: "${SETV_ULA_SMOKE_MODE:?missing SETV_ULA_SMOKE_MODE}"
: "${SETV_ULA_SSL_CHECKPOINT:?missing SETV_ULA_SSL_CHECKPOINT}"

MAIN_ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
ROOT=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95
SMOKE_ROOT="${ROOT}/preflight/ula_smoke_job${SLURM_JOB_ID}"
COMPATIBILITY_REPORT="${SMOKE_ROOT}/compatibility.json"
PROXY_REPORT="${SMOKE_ROOT}/proxy_checkpoint_smoke.json"
FINAL_REPORT="${SMOKE_ROOT}/smoke_receipt.json"
mkdir -p "$SMOKE_ROOT"

record_failure() {
  local status=$?
  local line=${1:-0}
  "${MAIN_ENV}/bin/python" scripts/record_ula_smoke_failure.py \
    --output "$FINAL_REPORT" \
    --mode "$SETV_ULA_SMOKE_MODE" \
    --exit-code "$status" \
    --line "$line" \
    --compatibility-report "$COMPATIBILITY_REPORT" \
    --proxy-report "$PROXY_REPORT" \
    --slurm-job-id "$SLURM_JOB_ID" || true
  exit "$status"
}
trap 'record_failure $LINENO' ERR

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "$SETV_REPO"
actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != "$SETV_EXPECTED_COMMIT" ]]; then
  echo "uLA smoke commit mismatch: expected=$SETV_EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "uLA smoke refuses a dirty checkout" >&2
  git status --short >&2
  exit 2
fi

if [[ "$SETV_ULA_SMOKE_MODE" == "official_ssl" ]]; then
  : "${SETV_ULA_ENV:?official SSL smoke requires SETV_ULA_ENV}"
  test -x "${SETV_ULA_ENV}/bin/python"
  "${SETV_ULA_ENV}/bin/python" scripts/check_ula_compatibility.py \
    --official-repo "$SETV_ULA_REPO" \
    --mode official_ssl \
    --require-gpu \
    --report "$COMPATIBILITY_REPORT"
  SETV_ULA_SHADOW_DIR="${SMOKE_ROOT}/shadow"
  SETV_ULA_SSL_DIR="${SMOKE_ROOT}/official_ssl"
  export SETV_ULA_SHADOW_DIR SETV_ULA_SSL_DIR
  "${MAIN_ENV}/bin/python" scripts/prepare_ula_official_shadow.py \
    --phase0-dir "${ROOT}/phase0" \
    --output-dir "$SETV_ULA_SHADOW_DIR" \
    --smoke-samples-per-split 8
  SETV_ULA_MAX_EPOCHS=1 \
  SETV_ULA_BATCH_SIZE=4 \
  SETV_ULA_NUM_WORKERS=0 \
  SETV_ULA_QUEUE_SIZE=8 \
  SETV_ULA_PRECISION=16 \
  SETV_ULA_CHECKPOINT_FREQUENCY=1 \
  SETV_ULA_RUN_NAME=setv_waterbirds95_official_mocov2plus_smoke \
  PATH="${SETV_ULA_ENV}/bin:${PATH}" \
    bash scripts/run_official_ula_ssl.sh
  SETV_ULA_SSL_CHECKPOINT="${SETV_ULA_SSL_DIR}/last.ckpt"
else
  if [[ "$SETV_ULA_SMOKE_MODE" != "external_checkpoint" ]]; then
    echo "Unknown SETV_ULA_SMOKE_MODE=$SETV_ULA_SMOKE_MODE" >&2
    exit 2
  fi
  test -f "$SETV_ULA_SSL_CHECKPOINT"
  "${MAIN_ENV}/bin/python" scripts/check_ula_compatibility.py \
    --official-repo "$SETV_ULA_REPO" \
    --mode external_checkpoint \
    --require-gpu \
    --report "$COMPATIBILITY_REPORT"
fi

test -f "$SETV_ULA_SSL_CHECKPOINT"
"${MAIN_ENV}/bin/python" scripts/smoke_ula_proxy.py \
  --config configs/ula_proxy_waterbirds95.yaml \
  --seed "$SETV_ULA_SEED" \
  --official-repo "$SETV_ULA_REPO" \
  --ssl-checkpoint "$SETV_ULA_SSL_CHECKPOINT" \
  --device cuda \
  --report "$PROXY_REPORT"
"${MAIN_ENV}/bin/python" scripts/finalize_ula_smoke.py \
  --mode "$SETV_ULA_SMOKE_MODE" \
  --compatibility-report "$COMPATIBILITY_REPORT" \
  --proxy-smoke-report "$PROXY_REPORT" \
  --official-repo "$SETV_ULA_REPO" \
  --ssl-checkpoint "$SETV_ULA_SSL_CHECKPOINT" \
  --setv-commit "$SETV_EXPECTED_COMMIT" \
  --output "$FINAL_REPORT"
echo "Accepted uLA smoke receipt: $FINAL_REPORT"
