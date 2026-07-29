#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/load_campaign_env.sh"

: "${SETV_ULA_SEED:?SETV_ULA_SEED must be explicitly frozen}"
: "${SETV_ULA_REPO:?SETV_ULA_REPO must be explicit}"
if [[ ! "$SETV_ULA_SEED" =~ ^[0-9]+$ ]]; then
  echo "SETV_ULA_SEED must be a nonnegative integer" >&2
  exit 2
fi

SETV_REPO=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ROOT=$SETV_CAMPAIGN_ROOT
if ! git -C "$SETV_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Refusing uLA smoke outside a Git checkout" >&2
  exit 2
fi
if [[ -n "$(git -C "$SETV_REPO" status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing uLA smoke from a dirty checkout" >&2
  git -C "$SETV_REPO" status --short >&2
  exit 2
fi
SETV_EXPECTED_COMMIT=$(git -C "$SETV_REPO" rev-parse HEAD)

if [[ -z "${SETV_ULA_SSL_CHECKPOINT:-}" ]]; then
  : "${SETV_ULA_ENV:?official SSL smoke requires SETV_ULA_ENV}"
  if [[ ! -x "${SETV_ULA_ENV}/bin/python" ]]; then
    echo "SETV_ULA_ENV does not contain an executable Python" >&2
    exit 2
  fi
  SETV_ULA_SMOKE_MODE=official_ssl
  # The smoke job replaces this placeholder with its one-epoch checkpoint.
  SETV_ULA_SSL_CHECKPOINT="${ROOT}/preflight/not_used_outside_smoke.ckpt"
else
  SETV_ULA_SSL_CHECKPOINT=$(realpath "$SETV_ULA_SSL_CHECKPOINT")
  test -f "$SETV_ULA_SSL_CHECKPOINT"
  SETV_ULA_SMOKE_MODE=external_checkpoint
fi

cd "$SETV_REPO"
python scripts/audit_ula_source.py --official-repo "$SETV_ULA_REPO" >/dev/null
test -f "${ROOT}/phase0/mask_audit/visual_review_approval.json" || {
  echo "Approved Phase 0 is required before the uLA data-path smoke" >&2
  exit 2
}
if [[ -n "$(squeue -h -u "$USER" -n setv_ula_smoke 2>/dev/null)" ]]; then
  echo "An active uLA smoke job already exists" >&2
  exit 2
fi
mkdir -p "${ROOT}/run_logs" "${ROOT}/submission_receipts" "${ROOT}/preflight"
export SETV_REPO SETV_EXPECTED_COMMIT SETV_ULA_REPO SETV_ULA_SEED
export SETV_ULA_ENV SETV_ULA_SSL_CHECKPOINT SETV_ULA_SMOKE_MODE
raw=$(sbatch --parsable --export=ALL \
  "${SETV_REPO}/slurm/phase6_ula_smoke.sbatch")
job_id=${raw%%;*}
receipt="${ROOT}/submission_receipts/phase6_ula_smoke_${job_id}.txt"
{
  echo "submitted_at=$(date --iso-8601=seconds)"
  echo "job_id=$job_id"
  echo "setv_commit=$SETV_EXPECTED_COMMIT"
  echo "mode=$SETV_ULA_SMOKE_MODE"
  echo "ula_repo=$SETV_ULA_REPO"
  echo "ula_environment=${SETV_ULA_ENV:-external_checkpoint_adapter}"
  echo "ssl_checkpoint=$SETV_ULA_SSL_CHECKPOINT"
  echo "expected_receipt=${ROOT}/preflight/ula_smoke_job${job_id}/smoke_receipt.json"
} > "$receipt"
echo "Submitted isolated uLA GH200 smoke $job_id"
echo "Expected result: ${ROOT}/preflight/ula_smoke_job${job_id}/smoke_receipt.json"
echo "Submission receipt: $receipt"
