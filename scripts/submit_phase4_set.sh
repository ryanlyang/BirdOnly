#!/usr/bin/env bash
set -Eeuo pipefail

for name in SETV_OBJECT_SEED SETV_SET_SEED SETV_FUSION_SEED; do
  if [[ -z "${!name:-}" || ! "${!name}" =~ ^[0-9]+$ ]]; then
    echo "$name must be an explicitly frozen nonnegative integer" >&2
    exit 2
  fi
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SETV_REPO=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ROOT=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95
PHASE0="${ROOT}/phase0"
OBJECT="${ROOT}/object_expert/seed_${SETV_OBJECT_SEED}"
SET_EXPERT="${ROOT}/background_set/seed_${SETV_SET_SEED}"
FUSION="${ROOT}/fusion_set/object_${SETV_OBJECT_SEED}_set_${SETV_SET_SEED}_fusion_${SETV_FUSION_SEED}"
mkdir -p "${ROOT}/run_logs" "${ROOT}/preflight" "${ROOT}/submission_receipts" \
  "${ROOT}/background_set" "${ROOT}/fusion_set"

if ! git -C "$SETV_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Refusing submission outside a Git checkout" >&2
  exit 2
fi
if [[ -n "$(git -C "$SETV_REPO" status --short)" ]]; then
  echo "Refusing submission from a dirty checkout" >&2
  git -C "$SETV_REPO" status --short >&2
  exit 2
fi
if [[ ! -f "${PHASE0}/mask_audit/visual_review_approval.json" ]]; then
  echo "Approved Phase 0 is missing: $PHASE0" >&2
  exit 2
fi
if [[ ! -f "${OBJECT}/phase1_receipt.json" ]]; then
  echo "Verified Phase 1 object expert is missing: $OBJECT" >&2
  exit 2
fi
if [[ -e "$SET_EXPERT" || -e "$FUSION" ]]; then
  echo "Refusing duplicate Phase 4 output" >&2
  exit 2
fi
if [[ -n "$(squeue -h -u "$USER" -n setv_set_smoke,setv_set_train,setv_set_fusion 2>/dev/null)" ]]; then
  echo "Refusing duplicate Phase 4 submission: active jobs exist" >&2
  exit 2
fi

export SETV_REPO SETV_OBJECT_SEED SETV_SET_SEED SETV_FUSION_SEED
smoke_raw=$(sbatch --parsable --export=ALL "${SETV_REPO}/slurm/phase4_set_smoke.sbatch")
smoke_id=${smoke_raw%%;*}
train_raw=$(sbatch --parsable --dependency="afterok:${smoke_id}" --export=ALL \
  "${SETV_REPO}/slurm/phase4_set_train.sbatch")
train_id=${train_raw%%;*}
fusion_raw=$(sbatch --parsable --dependency="afterok:${train_id}" --export=ALL \
  "${SETV_REPO}/slurm/phase4_set_fusion.sbatch")
fusion_id=${fusion_raw%%;*}

commit=$(git -C "$SETV_REPO" rev-parse HEAD)
receipt="${ROOT}/submission_receipts/phase4_set_${fusion_id}.txt"
{
  echo "submitted_at=$(date --iso-8601=seconds)"
  echo "object_seed=$SETV_OBJECT_SEED"
  echo "set_seed=$SETV_SET_SEED"
  echo "fusion_seed=$SETV_FUSION_SEED"
  echo "smoke_job_id=$smoke_id"
  echo "train_job_id=$train_id"
  echo "fusion_job_id=$fusion_id"
  echo "dependencies=${smoke_id}->${train_id}->${fusion_id}"
  echo "commit=$commit"
  echo "phase0_manifest_sha256=$(sha256sum "${PHASE0}/artifact_manifest.json" | awk '{print $1}')"
  echo "object_scores_sha256=$(sha256sum "${OBJECT}/scores/object_val_scores.npz" | awk '{print $1}')"
  echo "expert_config_sha256=$(sha256sum "${SETV_REPO}/configs/expert_background_set.yaml" | awk '{print $1}')"
  echo "fusion_config_sha256=$(sha256sum "${SETV_REPO}/configs/fusion_set.yaml" | awk '{print $1}')"
  echo "expected_set_output=$SET_EXPERT"
  echo "expected_fusion_output=$FUSION"
} > "$receipt"
echo "Submitted Phase 4: smoke=$smoke_id train=$train_id fusion=$fusion_id"
echo "Receipt: $receipt"
