#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/load_campaign_env.sh"

required=(
  SETV_CANDIDATE_SEEDS
  SETV_ULA_SEED
  SETV_ULA_REPO
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
IFS=',' read -r -a candidate_seeds <<< "$SETV_CANDIDATE_SEEDS"
if (( ${#candidate_seeds[@]} < 3 )); then
  echo "Phase 6 requires at least three candidate seeds" >&2
  exit 2
fi
declare -A seen=()
for seed in "${candidate_seeds[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]] || [[ -n "${seen[$seed]:-}" ]]; then
    echo "Candidate seeds must be unique nonnegative integers" >&2
    exit 2
  fi
  seen[$seed]=1
done
if [[ ! "$SETV_ULA_SEED" =~ ^[0-9]+$ ]]; then
  echo "SETV_ULA_SEED must be a nonnegative integer" >&2
  exit 2
fi

SETV_REPO=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ROOT=$SETV_CAMPAIGN_ROOT
SETV_ULA_SHADOW_DIR="${ROOT}/ula_official_shadow"
SETV_ULA_SSL_DIR="${ROOT}/ula_official_ssl"
if [[ -z "${SETV_ULA_SSL_CHECKPOINT:-}" ]]; then
  SETV_ULA_SSL_CHECKPOINT="${SETV_ULA_SSL_DIR}/last.ckpt"
  run_ssl=1
else
  SETV_ULA_SSL_CHECKPOINT=$(realpath "$SETV_ULA_SSL_CHECKPOINT")
  run_ssl=0
fi
SETV_ULA_PROXY_DIR="${ROOT}/ula_proxy/seed_${SETV_ULA_SEED}"
SETV_EXACT_FUSION_DIR="${ROOT}/fusion_exact/object_${SETV_OBJECT_SEED}_exact_${SETV_EXACT_SEED}_fusion_${SETV_EXACT_FUSION_SEED}"
SETV_SANITIZED_FUSION_DIR="${ROOT}/fusion_sanitized/object_${SETV_OBJECT_SEED}_sanitized_${SETV_SANITIZED_SEED}_fusion_${SETV_SANITIZED_FUSION_SEED}"
SETV_SET_FUSION_DIR="${ROOT}/fusion_set/object_${SETV_OBJECT_SEED}_set_${SETV_SET_SEED}_fusion_${SETV_SET_FUSION_SEED}"
export SETV_REPO SETV_ULA_SHADOW_DIR SETV_ULA_SSL_DIR
export SETV_ULA_SSL_CHECKPOINT SETV_ULA_PROXY_DIR
export SETV_EXACT_FUSION_DIR SETV_SANITIZED_FUSION_DIR SETV_SET_FUSION_DIR
export SETV_ULA_SEED SETV_ULA_REPO SETV_ULA_ENV

if ! git -C "$SETV_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Refusing submission outside a Git checkout" >&2
  exit 2
fi
if [[ -n "$(git -C "$SETV_REPO" status --short)" ]]; then
  echo "Refusing submission from a dirty checkout" >&2
  exit 2
fi
SETV_EXPECTED_COMMIT=$(git -C "$SETV_REPO" rev-parse HEAD)
export SETV_EXPECTED_COMMIT
cd "$SETV_REPO"
if (( run_ssl )); then
  if [[ -z "${SETV_ULA_ENV:-}" ]] || [[ ! -x "${SETV_ULA_ENV}/bin/python" ]]; then
    echo "SETV_ULA_ENV is required to run official SSL and must be usable" >&2
    exit 2
  fi
  SETV_ULA_SMOKE_MODE=official_ssl
else
  if [[ ! -f "$SETV_ULA_SSL_CHECKPOINT" ]]; then
    echo "Explicit SETV_ULA_SSL_CHECKPOINT does not exist" >&2
    exit 2
  fi
  SETV_ULA_SMOKE_MODE=external_checkpoint
fi
export SETV_ULA_SMOKE_MODE
python scripts/audit_ula_source.py --official-repo "$SETV_ULA_REPO" >/dev/null
for path in "$SETV_EXACT_FUSION_DIR" "$SETV_SANITIZED_FUSION_DIR" "$SETV_SET_FUSION_DIR"; do
  test -f "${path}/fusion_receipt.json" || {
    echo "Missing fusion artifact: $path" >&2
    exit 2
  }
done
for seed in "${candidate_seeds[@]}"; do
  test -f "${ROOT}/candidate_erm/seed_${seed}/phase5_receipt.json" || {
    echo "Missing Phase 5 candidate seed: $seed" >&2
    exit 2
  }
done
outputs=("$SETV_ULA_PROXY_DIR" "${ROOT}/phase6")
if (( run_ssl )); then
  outputs+=("$SETV_ULA_SHADOW_DIR" "$SETV_ULA_SSL_DIR")
fi
for path in "${outputs[@]}"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite Phase 6 output: $path" >&2
    exit 2
  fi
done
mkdir -p "${ROOT}/run_logs" "${ROOT}/preflight" \
  "${ROOT}/submission_receipts" "${ROOT}/ula_proxy"

preflight_report="${ROOT}/preflight/phase6_submission_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}.json"
bash "${SCRIPT_DIR}/run_submission_preflight.sh" phase6 "$preflight_report"
preflight_sha=$(sha256sum "$preflight_report" | awk '{print $1}')

smoke_raw=$(sbatch --parsable --export=ALL \
  "${SETV_REPO}/slurm/phase6_ula_smoke.sbatch")
smoke_id=${smoke_raw%%;*}
if (( run_ssl )); then
  ssl_raw=$(sbatch --parsable --dependency="afterok:${smoke_id}" --export=ALL \
    "${SETV_REPO}/slurm/phase6_ula_ssl.sbatch")
  ssl_id=${ssl_raw%%;*}
  proxy_raw=$(sbatch --parsable --dependency="afterok:${ssl_id}" --export=ALL \
    "${SETV_REPO}/slurm/phase6_ula_proxy.sbatch")
  ssl_dependency="afterok:${smoke_id}"
  proxy_dependency="afterok:${ssl_id}"
else
  ssl_id="external_checkpoint"
  ssl_dependency="not_applicable"
  proxy_raw=$(sbatch --parsable --dependency="afterok:${smoke_id}" --export=ALL \
    "${SETV_REPO}/slurm/phase6_ula_proxy.sbatch")
  proxy_dependency="afterok:${smoke_id}"
fi
proxy_id=${proxy_raw%%;*}
analysis_raw=$(sbatch --parsable --dependency="afterok:${proxy_id}" --export=ALL \
  "${SETV_REPO}/slurm/phase6_analysis.sbatch")
analysis_id=${analysis_raw%%;*}

commit=$SETV_EXPECTED_COMMIT
receipt="${ROOT}/submission_receipts/phase6_${ssl_id}.txt"
{
  echo "submitted_at=$(date --iso-8601=seconds)"
  echo "commit=$commit"
  echo "campaign_manifest=$SETV_CAMPAIGN_CONFIG"
  echo "preflight_report=$preflight_report"
  echo "preflight_sha256=$preflight_sha"
  echo "candidate_seeds=$SETV_CANDIDATE_SEEDS"
  echo "candidate_seed_count=${#candidate_seeds[@]}"
  echo "ula_repo=$SETV_ULA_REPO"
  echo "ula_official_commit=5867fb6e9a8485ed08b4cbe84900f2b5ac4fac5d"
  echo "ula_environment=${SETV_ULA_ENV:-external_checkpoint_not_trained_here}"
  echo "ula_ssl_checkpoint=$SETV_ULA_SSL_CHECKPOINT"
  echo "ula_smoke_mode=$SETV_ULA_SMOKE_MODE"
  echo "ula_smoke_job_id=$smoke_id"
  echo "ula_smoke_receipt=${ROOT}/preflight/ula_smoke_job${smoke_id}/smoke_receipt.json"
  echo "ssl_job_id=$ssl_id"
  echo "proxy_job_id=$proxy_id"
  echo "analysis_job_id=$analysis_id"
  echo "ssl_dependency=$ssl_dependency"
  echo "proxy_dependency=$proxy_dependency"
  echo "analysis_dependency=afterok:${proxy_id}"
} > "$receipt"
echo "Submitted Phase 6 smoke=$smoke_id SSL=$ssl_id proxy=$proxy_id analysis=$analysis_id"
echo "Receipt: $receipt"
