#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/load_campaign_env.sh"

if [[ -z "${SETV_SET_SEED:-}" || ! "${SETV_SET_SEED}" =~ ^[0-9]+$ ]]; then
  echo "SETV_SET_SEED must be an explicitly frozen nonnegative integer" >&2
  exit 2
fi

SETV_REPO=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ROOT=$SETV_CAMPAIGN_ROOT
PHASE0="${ROOT}/phase0"
mkdir -p "${ROOT}/run_logs" "${ROOT}/preflight" \
  "${ROOT}/submission_receipts"

if ! git -C "$SETV_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Refusing capacity audit outside a Git checkout" >&2
  exit 2
fi
if [[ -n "$(git -C "$SETV_REPO" status --short)" ]]; then
  echo "Refusing capacity audit from a dirty checkout" >&2
  git -C "$SETV_REPO" status --short >&2
  exit 2
fi
if [[ ! -f "${PHASE0}/mask_audit/visual_review_approval.json" ]]; then
  echo "Approved Phase 0 is missing: $PHASE0" >&2
  exit 2
fi
if [[ -n "$(squeue -h -u "$USER" -n setv_set_capacity 2>/dev/null)" ]]; then
  echo "Refusing duplicate capacity audit: an active job exists" >&2
  exit 2
fi

preflight_report="${ROOT}/preflight/phase4_capacity_submission_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.json"
bash "${SCRIPT_DIR}/run_submission_preflight.sh" phase4 "$preflight_report"
preflight_sha=$(sha256sum "$preflight_report" | awk '{print $1}')

SETV_EXPECTED_COMMIT=$(git -C "$SETV_REPO" rev-parse HEAD)
export SETV_REPO SETV_EXPECTED_COMMIT SETV_SET_SEED
job_raw=$(sbatch --parsable --export=ALL \
  "${SETV_REPO}/slurm/phase4_set_capacity.sbatch")
job_id=${job_raw%%;*}
receipt="${ROOT}/submission_receipts/phase4_set_capacity_${job_id}.txt"
{
  echo "submitted_at=$(date --iso-8601=seconds)"
  echo "job_id=$job_id"
  echo "set_seed=$SETV_SET_SEED"
  echo "commit=$SETV_EXPECTED_COMMIT"
  echo "campaign_manifest=$SETV_CAMPAIGN_CONFIG"
  echo "preflight_report=$preflight_report"
  echo "preflight_sha256=$preflight_sha"
  echo "phase0_manifest_sha256=$(sha256sum "${PHASE0}/artifact_manifest.json" | awk '{print $1}')"
  echo "expert_config_sha256=$(sha256sum "${SETV_REPO}/configs/expert_background_set.yaml" | awk '{print $1}')"
  echo "expected_json_report=${ROOT}/preflight/set_capacity_job${job_id}.json"
  echo "expected_csv_report=${ROOT}/preflight/set_capacity_samples_job${job_id}.csv"
} > "$receipt"

echo "Submitted Stage 4 capacity audit: job=$job_id"
echo "Receipt: $receipt"
