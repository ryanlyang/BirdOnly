#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/load_campaign_env.sh"

for name in SETV_OBJECT_SEED SETV_EXACT_SEED SETV_EXACT_FUSION_SEED; do
  if [[ -z "${!name:-}" || ! "${!name}" =~ ^[0-9]+$ ]]; then
    echo "$name must be an explicitly frozen nonnegative integer" >&2
    exit 2
  fi
done

SETV_REPO=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ROOT=$SETV_CAMPAIGN_ROOT
OBJECT="${ROOT}/object_expert/seed_${SETV_OBJECT_SEED}"
EXACT="${ROOT}/background_exact/seed_${SETV_EXACT_SEED}"
FUSION="${ROOT}/fusion_exact/object_${SETV_OBJECT_SEED}_exact_${SETV_EXACT_SEED}_fusion_${SETV_EXACT_FUSION_SEED}"
mkdir -p "${ROOT}/run_logs" "${ROOT}/preflight" "${ROOT}/submission_receipts" \
  "${ROOT}/background_exact" "${ROOT}/fusion_exact"

if ! git -C "$SETV_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Refusing submission outside a Git checkout" >&2
  exit 2
fi
if [[ -n "$(git -C "$SETV_REPO" status --short)" ]]; then
  echo "Refusing submission from a dirty checkout" >&2
  git -C "$SETV_REPO" status --short >&2
  exit 2
fi
if [[ ! -f "${OBJECT}/phase1_receipt.json" ]]; then
  echo "Verified Phase 1 object expert is missing: $OBJECT" >&2
  exit 2
fi
if [[ -e "$EXACT" || -e "$FUSION" ]]; then
  echo "Refusing duplicate Phase 2 output" >&2
  exit 2
fi
if [[ -n "$(squeue -h -u "$USER" -n setv_exact_smoke,setv_exact_train,setv_exact_fusion 2>/dev/null)" ]]; then
  echo "Refusing duplicate Phase 2 submission: active jobs exist" >&2
  exit 2
fi

preflight_report="${ROOT}/preflight/phase2_submission_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.json"
bash "${SCRIPT_DIR}/run_submission_preflight.sh" phase2 "$preflight_report"
preflight_sha=$(sha256sum "$preflight_report" | awk '{print $1}')

export SETV_REPO SETV_OBJECT_SEED SETV_EXACT_SEED SETV_EXACT_FUSION_SEED
smoke_raw=$(sbatch --parsable --export=ALL "${SETV_REPO}/slurm/phase2_exact_smoke.sbatch")
smoke_id=${smoke_raw%%;*}
train_raw=$(sbatch --parsable --dependency="afterok:${smoke_id}" --export=ALL \
  "${SETV_REPO}/slurm/phase2_exact_train.sbatch")
train_id=${train_raw%%;*}
fusion_raw=$(sbatch --parsable --dependency="afterok:${train_id}" --export=ALL \
  "${SETV_REPO}/slurm/phase2_exact_fusion.sbatch")
fusion_id=${fusion_raw%%;*}

commit=$(git -C "$SETV_REPO" rev-parse HEAD)
receipt="${ROOT}/submission_receipts/phase2_exact_${fusion_id}.txt"
{
  echo "submitted_at=$(date --iso-8601=seconds)"
  echo "object_seed=$SETV_OBJECT_SEED"
  echo "exact_seed=$SETV_EXACT_SEED"
  echo "fusion_seed=$SETV_EXACT_FUSION_SEED"
  echo "smoke_job_id=$smoke_id"
  echo "train_job_id=$train_id"
  echo "fusion_job_id=$fusion_id"
  echo "dependencies=${smoke_id}->${train_id}->${fusion_id}"
  echo "commit=$commit"
  echo "campaign_manifest=$SETV_CAMPAIGN_CONFIG"
  echo "preflight_report=$preflight_report"
  echo "preflight_sha256=$preflight_sha"
  echo "object_scores_sha256=$(sha256sum "${OBJECT}/scores/object_val_scores.npz" | awk '{print $1}')"
  echo "exact_config_sha256=$(sha256sum "${SETV_REPO}/configs/expert_background_exact.yaml" | awk '{print $1}')"
  echo "fusion_config_sha256=$(sha256sum "${SETV_REPO}/configs/fusion_exact.yaml" | awk '{print $1}')"
  echo "expected_exact_output=$EXACT"
  echo "expected_fusion_output=$FUSION"
} > "$receipt"
echo "Submitted Phase 2: smoke=$smoke_id train=$train_id fusion=$fusion_id"
echo "Receipt: $receipt"
