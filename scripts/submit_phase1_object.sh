#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/load_campaign_env.sh"

: "${SETV_OBJECT_SEED:?Set SETV_OBJECT_SEED to the explicitly frozen integer seed}"
if [[ ! "$SETV_OBJECT_SEED" =~ ^[0-9]+$ ]]; then
  echo "SETV_OBJECT_SEED must be a nonnegative integer" >&2
  exit 2
fi

SETV_REPO=$(cd -- "${SCRIPT_DIR}/.." && pwd)
CAMPAIGN_ROOT=$SETV_CAMPAIGN_ROOT
PHASE0_DIR="${CAMPAIGN_ROOT}/phase0"
OUTPUT_DIR="${CAMPAIGN_ROOT}/object_expert/seed_${SETV_OBJECT_SEED}"
LOG_DIR="${CAMPAIGN_ROOT}/run_logs"
PREFLIGHT_DIR="${CAMPAIGN_ROOT}/preflight"
RECEIPT_DIR="${CAMPAIGN_ROOT}/submission_receipts"

mkdir -p "$LOG_DIR" "$PREFLIGHT_DIR" "$RECEIPT_DIR" "${CAMPAIGN_ROOT}/object_expert"

if ! git -C "$SETV_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Refusing submission: SETV_REPO is not a Git checkout: $SETV_REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$SETV_REPO" status --short)" ]]; then
  echo "Refusing submission from a dirty checkout:" >&2
  git -C "$SETV_REPO" status --short >&2
  exit 2
fi
if [[ ! -f "${PHASE0_DIR}/mask_audit/visual_review_approval.json" ]]; then
  echo "Refusing submission: approved Phase 0 is missing" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing duplicate object-expert run: $OUTPUT_DIR exists" >&2
  exit 2
fi
if [[ -n "$(squeue -h -u "$USER" -n setv_obj_smoke,setv_obj_train 2>/dev/null)" ]]; then
  echo "Refusing duplicate submission: an object-expert job is active" >&2
  exit 2
fi

preflight_report="${PREFLIGHT_DIR}/phase1_submission_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.json"
bash "${SCRIPT_DIR}/run_submission_preflight.sh" phase1 "$preflight_report"
preflight_sha=$(sha256sum "$preflight_report" | awk '{print $1}')

commit=$(git -C "$SETV_REPO" rev-parse HEAD)
config_sha=$(sha256sum "${SETV_REPO}/configs/expert_object_green.yaml" | awk '{print $1}')
plan_sha=$(sha256sum "${SETV_REPO}/SETV_Waterbirds95_Implementation_Plan_v2.md" | awk '{print $1}')
phase0_manifest_sha=$(sha256sum "${PHASE0_DIR}/artifact_manifest.json" | awk '{print $1}')
phase0_approval_sha=$(sha256sum "${PHASE0_DIR}/mask_audit/visual_review_approval.json" | awk '{print $1}')

smoke_raw=$(sbatch --parsable \
  --export="ALL,SETV_REPO=${SETV_REPO},SETV_OBJECT_SEED=${SETV_OBJECT_SEED}" \
  "${SETV_REPO}/slurm/phase1_object_smoke.sbatch")
smoke_id=${smoke_raw%%;*}
train_raw=$(sbatch --parsable \
  --dependency="afterok:${smoke_id}" \
  --export="ALL,SETV_REPO=${SETV_REPO},SETV_OBJECT_SEED=${SETV_OBJECT_SEED}" \
  "${SETV_REPO}/slurm/phase1_object_train.sbatch")
train_id=${train_raw%%;*}

receipt="${RECEIPT_DIR}/phase1_object_seed${SETV_OBJECT_SEED}_${train_id}.txt"
{
  echo "submitted_at=$(date --iso-8601=seconds)"
  echo "seed=$SETV_OBJECT_SEED"
  echo "smoke_job_id=$smoke_id"
  echo "train_job_id=$train_id"
  echo "dependency=afterok:${smoke_id}"
  echo "commit=$commit"
  echo "repo=$SETV_REPO"
  echo "campaign_manifest=$SETV_CAMPAIGN_CONFIG"
  echo "preflight_report=$preflight_report"
  echo "preflight_sha256=$preflight_sha"
  echo "config_sha256=$config_sha"
  echo "implementation_plan_sha256=$plan_sha"
  echo "phase0_manifest_sha256=$phase0_manifest_sha"
  echo "phase0_visual_approval_sha256=$phase0_approval_sha"
  echo "expected_output=$OUTPUT_DIR"
} > "$receipt"

echo "Submitted object-expert smoke ${smoke_id}"
echo "Submitted object-expert training ${train_id} afterok:${smoke_id}"
echo "Receipt: ${receipt}"
