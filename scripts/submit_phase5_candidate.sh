#!/usr/bin/env bash
set -Eeuo pipefail

required=(
  SETV_CANDIDATE_SEEDS
  SETV_OBJECT_SEED
  SETV_EXACT_SEED
  SETV_SANITIZED_SEED
  SETV_SET_SEED
  SETV_EXACT_FUSION_SEED
  SETV_SANITIZED_FUSION_SEED
  SETV_SET_FUSION_SEED
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name must be explicitly frozen" >&2
    exit 2
  fi
done
for name in SETV_OBJECT_SEED SETV_EXACT_SEED SETV_SANITIZED_SEED SETV_SET_SEED \
  SETV_EXACT_FUSION_SEED SETV_SANITIZED_FUSION_SEED SETV_SET_FUSION_SEED; do
  if [[ ! "${!name}" =~ ^[0-9]+$ ]]; then
    echo "$name must be a nonnegative integer" >&2
    exit 2
  fi
done

IFS=',' read -r -a candidate_seeds <<< "$SETV_CANDIDATE_SEEDS"
declare -A seen_seeds=()
for seed in "${candidate_seeds[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "SETV_CANDIDATE_SEEDS must be comma-separated nonnegative integers" >&2
    exit 2
  fi
  if [[ -n "${seen_seeds[$seed]:-}" ]]; then
    echo "Candidate seeds must be unique: $seed" >&2
    exit 2
  fi
  seen_seeds[$seed]=1
done
if (( ${#candidate_seeds[@]} < 3 )); then
  echo "The private pilot requires at least three ERM candidate seeds" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SETV_REPO=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ROOT=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95
SETV_EXACT_FUSION_DIR="${ROOT}/fusion_exact/object_${SETV_OBJECT_SEED}_exact_${SETV_EXACT_SEED}_fusion_${SETV_EXACT_FUSION_SEED}"
SETV_SANITIZED_FUSION_DIR="${ROOT}/fusion_sanitized/object_${SETV_OBJECT_SEED}_sanitized_${SETV_SANITIZED_SEED}_fusion_${SETV_SANITIZED_FUSION_SEED}"
SETV_SET_FUSION_DIR="${ROOT}/fusion_set/object_${SETV_OBJECT_SEED}_set_${SETV_SET_SEED}_fusion_${SETV_SET_FUSION_SEED}"
mkdir -p "${ROOT}/run_logs" "${ROOT}/preflight" \
  "${ROOT}/submission_receipts" "${ROOT}/candidate_erm"

if ! git -C "$SETV_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Refusing submission outside a Git checkout" >&2
  exit 2
fi
if [[ -n "$(git -C "$SETV_REPO" status --short)" ]]; then
  echo "Refusing submission from a dirty checkout" >&2
  git -C "$SETV_REPO" status --short >&2
  exit 2
fi
for path in "$SETV_EXACT_FUSION_DIR" "$SETV_SANITIZED_FUSION_DIR" "$SETV_SET_FUSION_DIR"; do
  if [[ ! -f "${path}/fusion_receipt.json" ]]; then
    echo "Required verified fusion is missing: $path" >&2
    exit 2
  fi
done
for seed in "${candidate_seeds[@]}"; do
  if [[ -e "${ROOT}/candidate_erm/seed_${seed}" ]]; then
    echo "Refusing duplicate candidate output for seed $seed" >&2
    exit 2
  fi
done
if [[ -n "$(squeue -h -u "$USER" -n setv_candidate_smoke,setv_candidate_train 2>/dev/null)" ]]; then
  echo "Refusing duplicate Phase 5 submission: active jobs exist" >&2
  exit 2
fi

SETV_SMOKE_CANDIDATE_SEED=${candidate_seeds[0]}
export SETV_REPO SETV_SMOKE_CANDIDATE_SEED
export SETV_EXACT_FUSION_DIR SETV_SANITIZED_FUSION_DIR SETV_SET_FUSION_DIR
smoke_raw=$(sbatch --parsable --export=ALL \
  "${SETV_REPO}/slurm/phase5_candidate_smoke.sbatch")
smoke_id=${smoke_raw%%;*}

train_ids=()
for seed in "${candidate_seeds[@]}"; do
  train_raw=$(sbatch --parsable --dependency="afterok:${smoke_id}" \
    --export="ALL,SETV_CANDIDATE_SEED=${seed}" \
    "${SETV_REPO}/slurm/phase5_candidate_train.sbatch")
  train_ids+=("${train_raw%%;*}")
done

commit=$(git -C "$SETV_REPO" rev-parse HEAD)
receipt="${ROOT}/submission_receipts/phase5_candidate_${smoke_id}.txt"
{
  echo "submitted_at=$(date --iso-8601=seconds)"
  echo "candidate_seeds=$SETV_CANDIDATE_SEEDS"
  echo "candidate_seed_count=${#candidate_seeds[@]}"
  echo "smoke_job_id=$smoke_id"
  echo "train_job_ids=${train_ids[*]}"
  echo "dependency=all trains afterok:${smoke_id}"
  echo "commit=$commit"
  echo "exact_fusion=$SETV_EXACT_FUSION_DIR"
  echo "sanitized_fusion=$SETV_SANITIZED_FUSION_DIR"
  echo "set_fusion=$SETV_SET_FUSION_DIR"
  echo "exact_fusion_receipt_sha256=$(sha256sum "${SETV_EXACT_FUSION_DIR}/fusion_receipt.json" | awk '{print $1}')"
  echo "sanitized_fusion_receipt_sha256=$(sha256sum "${SETV_SANITIZED_FUSION_DIR}/fusion_receipt.json" | awk '{print $1}')"
  echo "set_fusion_receipt_sha256=$(sha256sum "${SETV_SET_FUSION_DIR}/fusion_receipt.json" | awk '{print $1}')"
  echo "candidate_config_sha256=$(sha256sum "${SETV_REPO}/configs/candidate_erm.yaml" | awk '{print $1}')"
} > "$receipt"
echo "Submitted Phase 5 smoke=$smoke_id candidate seeds=$SETV_CANDIDATE_SEEDS"
echo "Train jobs: ${train_ids[*]}"
echo "Receipt: $receipt"
