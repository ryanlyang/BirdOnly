#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SETV_REPO=$(cd -- "${SCRIPT_DIR}/.." && pwd)
LOG_DIR=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs
RECEIPT_DIR=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/submission_receipts

mkdir -p "$LOG_DIR" "$RECEIPT_DIR"

if [[ -e /home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/phase0 ]]; then
  echo "Refusing duplicate Phase 0 build: output directory already exists" >&2
  exit 2
fi

if [[ -n "$(squeue -h -u "$USER" -n setv_wb95_phase0 2>/dev/null)" ]]; then
  echo "Refusing duplicate Phase 0 submission: an active job already exists" >&2
  exit 2
fi

commit=$(git -C "$SETV_REPO" rev-parse HEAD 2>/dev/null || printf UNKNOWN)
config_sha=$(sha256sum "${SETV_REPO}/configs/data_waterbirds95.yaml" | awk '{print $1}')
plan_sha=$(sha256sum "${SETV_REPO}/SETV_Waterbirds95_Implementation_Plan_v2.md" | awk '{print $1}')
handoff_sha=$(sha256sum "${SETV_REPO}/TIGRIS_RESEARCH_COMPUTE_HANDOFF.md" | awk '{print $1}')
job_id=$(sbatch --parsable --export="ALL,SETV_REPO=${SETV_REPO}" \
  "${SETV_REPO}/slurm/phase0_preflight.sbatch")
job_id=${job_id%%;*}
receipt="${RECEIPT_DIR}/phase0_submission_${job_id}.txt"
{
  echo "submitted_at=$(date --iso-8601=seconds)"
  echo "job_id=$job_id"
  echo "commit=$commit"
  echo "repo=$SETV_REPO"
  echo "config=${SETV_REPO}/configs/data_waterbirds95.yaml"
  echo "config_sha256=$config_sha"
  echo "implementation_plan_sha256=$plan_sha"
  echo "handoff_sha256=$handoff_sha"
} > "$receipt"

echo "Submitted Phase 0 job ${job_id}"
echo "Receipt: ${receipt}"
