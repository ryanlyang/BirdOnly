#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SETV_REPO=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/load_campaign_env.sh"
LOG_DIR="${SETV_CAMPAIGN_ROOT}/run_logs"
PREFLIGHT_DIR="${SETV_CAMPAIGN_ROOT}/preflight"
RECEIPT_DIR="${SETV_CAMPAIGN_ROOT}/submission_receipts"

mkdir -p "$LOG_DIR" "$PREFLIGHT_DIR" "$RECEIPT_DIR"

if ! git -C "$SETV_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Refusing Phase 0 submission: SETV_REPO is not a Git checkout" >&2
  exit 2
fi

if [[ -n "$(git -C "$SETV_REPO" status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing Phase 0 submission: Git worktree is not clean" >&2
  git -C "$SETV_REPO" status --short >&2
  exit 2
fi

if [[ -e "${SETV_CAMPAIGN_ROOT}/phase0" ]]; then
  echo "Refusing duplicate Phase 0 build: output directory already exists" >&2
  exit 2
fi

if [[ -n "$(squeue -h -u "$USER" -n setv_wb95_phase0 2>/dev/null)" ]]; then
  echo "Refusing duplicate Phase 0 submission: an active job already exists" >&2
  exit 2
fi

preflight_report="${PREFLIGHT_DIR}/phase0_submission_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.json"
bash "${SCRIPT_DIR}/run_submission_preflight.sh" phase0 "$preflight_report"
preflight_sha=$(sha256sum "$preflight_report" | awk '{print $1}')

commit=$(git -C "$SETV_REPO" rev-parse HEAD)
config_sha=$(sha256sum "${SETV_REPO}/configs/data_waterbirds95.yaml" | awk '{print $1}')
plan_sha=$(sha256sum "${SETV_REPO}/SETV_Waterbirds95_Implementation_Plan_v2.md" | awk '{print $1}')
handoff_sha=$(sha256sum "${SETV_REPO}/TIGRIS_RESEARCH_COMPUTE_HANDOFF.md" | awk '{print $1}')
job_id=$(sbatch --parsable \
  --export="ALL,SETV_REPO=${SETV_REPO},SETV_EXPECTED_COMMIT=${commit}" \
  "${SETV_REPO}/slurm/phase0_preflight.sbatch")
job_id=${job_id%%;*}
receipt="${RECEIPT_DIR}/phase0_submission_${job_id}.txt"
{
  echo "submitted_at=$(date --iso-8601=seconds)"
  echo "job_id=$job_id"
  echo "commit=$commit"
  echo "repo=$SETV_REPO"
  echo "campaign_manifest=$SETV_CAMPAIGN_CONFIG"
  echo "preflight_report=$preflight_report"
  echo "preflight_sha256=$preflight_sha"
  echo "config=${SETV_REPO}/configs/data_waterbirds95.yaml"
  echo "config_sha256=$config_sha"
  echo "implementation_plan_sha256=$plan_sha"
  echo "handoff_sha256=$handoff_sha"
  echo "expected_mask_mapping_audit=${SETV_CAMPAIGN_ROOT}/preflight/phase0_mask_mapping_job${job_id}.json"
  echo "expected_mask_galleries=${SETV_CAMPAIGN_ROOT}/preflight/phase0_mask_galleries_job${job_id}"
} > "$receipt"

echo "Submitted Phase 0 job ${job_id}"
echo "Receipt: ${receipt}"
