#!/usr/bin/env bash
# Shared fail-closed runtime setup for the AnchorCal TIGRIS jobs.
#
# Every .sbatch file verifies EXPECTED_COMMIT and the clean worktree before it
# sources this file.  This file then authenticates the frozen campaign inputs
# and Python environment before a project Python module can be imported.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "runtime_common.sh must be sourced by an AnchorCal batch job" >&2
  exit 2
fi

anchorcal_die() {
  echo "AnchorCal runtime failure: $*" >&2
  exit 2
}

anchorcal_require_var() {
  local name=$1
  [[ -n "${!name:-}" ]] || anchorcal_die "required environment variable ${name} is missing"
}

anchorcal_sha256() {
  local path=$1
  local digest ignored
  read -r digest ignored < <(sha256sum -- "$path")
  printf '%s\n' "$digest"
}

anchorcal_verify_file() {
  local label=$1
  local path=$2
  local expected=$3
  [[ -f "$path" ]] || anchorcal_die "${label} is missing: ${path}"
  local actual
  actual=$(anchorcal_sha256 "$path")
  [[ "$actual" == "$expected" ]] || \
    anchorcal_die "${label} hash changed (expected ${expected}, found ${actual}): ${path}"
}

anchorcal_verify_package_lock() {
  local temporary
  temporary=$(mktemp "${TMPDIR:-/tmp}/anchorcal-package-lock.${SLURM_JOB_ID:-nojob}.XXXXXX")
  "$ANCHORCAL_PYTHON" -c 'import sys
from anchorcal.runtime import write_package_lock
write_package_lock(sys.argv[1])' "$temporary"
  if ! cmp -s -- "$ANCHORCAL_PACKAGE_LOCK" "$temporary"; then
    echo "Frozen and current package locks differ:" >&2
    diff -u -- "$ANCHORCAL_PACKAGE_LOCK" "$temporary" >&2 || true
    rm -f -- "$temporary"
    anchorcal_die "the frozen Python environment changed after submission"
  fi
  rm -f -- "$temporary"
}

anchorcal_verify_decision_receipt() {
  local receipt_root="$ANCHORCAL_OUTPUT_ROOT/receipt"
  local receipts=()
  shopt -s nullglob
  receipts=("$receipt_root"/anchorcal_decision_*.json)
  shopt -u nullglob
  (( ${#receipts[@]} == 1 )) || \
    anchorcal_die "expected one AnchorCal decision receipt, found ${#receipts[@]} under ${receipt_root}"
  local receipt=${receipts[0]}
  local sidecar="${receipt}.sha256"
  [[ -f "$sidecar" ]] || anchorcal_die "decision receipt sidecar is missing: ${sidecar}"
  (
    cd -- "$receipt_root"
    sha256sum -c -- "$(basename -- "$sidecar")"
  ) >/dev/null || anchorcal_die "decision receipt hash verification failed: ${receipt}"
  printf '%s\n' "$receipt"
}

anchorcal_write_job_receipt() {
  local stage=$1
  local network_mode=$2
  local receipt_root="$ANCHORCAL_OUTPUT_ROOT/submission_receipts/jobs"
  mkdir -p -- "$receipt_root"
  local attempt=${SLURM_RESTART_COUNT:-0}
  [[ "$attempt" =~ ^[0-9]+$ ]] || anchorcal_die "invalid SLURM_RESTART_COUNT: ${attempt}"
  local receipt="$receipt_root/${stage}_job${SLURM_JOB_ID:-unknown}_attempt${attempt}.txt"
  [[ ! -e "$receipt" && ! -e "${receipt}.sha256" ]] || \
    anchorcal_die "job-attempt receipt already exists: ${receipt}"
  local temporary="${receipt}.partial.${BASHPID}"
  {
    printf 'schema_version=anchorcal-tigris-job-v1\n'
    printf 'stage=%s\n' "$stage"
    printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-not-set}"
    printf 'slurm_job_name=%s\n' "${SLURM_JOB_NAME:-not-set}"
    printf 'slurm_restart_count=%s\n' "$attempt"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'architecture=%s\n' "$(uname -m)"
    printf 'network_mode=%s\n' "$network_mode"
    printf 'repo=%s\n' "$ANCHORCAL_REPO"
    printf 'expected_commit=%s\n' "$ANCHORCAL_EXPECTED_COMMIT"
    printf 'python=%s\n' "$ANCHORCAL_PYTHON"
    printf 'python_sha256=%s\n' "$ANCHORCAL_PYTHON_SHA256"
    printf 'config=%s\n' "$ANCHORCAL_CONFIG"
    printf 'config_sha256=%s\n' "$ANCHORCAL_CONFIG_SHA256"
    printf 'paths=%s\n' "$ANCHORCAL_PATHS"
    printf 'paths_sha256=%s\n' "$ANCHORCAL_PATHS_SHA256"
    printf 'package_lock=%s\n' "$ANCHORCAL_PACKAGE_LOCK"
    printf 'package_lock_sha256=%s\n' "$ANCHORCAL_PACKAGE_LOCK_SHA256"
    printf 'frozen_input_receipt=%s\n' "$ANCHORCAL_INPUT_RECEIPT"
    printf 'frozen_input_receipt_sha256=%s\n' "$ANCHORCAL_INPUT_RECEIPT_SHA256"
    printf 'pythonhashseed=%s\n' "$PYTHONHASHSEED"
    printf 'cublas_workspace_config=%s\n' "$CUBLAS_WORKSPACE_CONFIG"
    printf 'hardware=%s\n' "${ANCHORCAL_HARDWARE_MODE:-unknown}"
    printf 'gpu=%s\n' "${ANCHORCAL_GPU_INFO:-none}"
    printf 'stdout_log=%s/run_logs/%s_%s.out\n' \
      "$ANCHORCAL_OUTPUT_ROOT" "${SLURM_JOB_NAME:-unknown}" "${SLURM_JOB_ID:-unknown}"
    printf 'stderr_log=%s/run_logs/%s_%s.err\n' \
      "$ANCHORCAL_OUTPUT_ROOT" "${SLURM_JOB_NAME:-unknown}" "${SLURM_JOB_ID:-unknown}"
    printf 'determinism_warning_policy=torch_warn_only_warnings_are_preserved_in_stderr_log\n'
    if [[ -n "${ANCHORCAL_CANDIDATE_TAG:-}" ]]; then
      printf 'candidate_tag=%s\n' "$ANCHORCAL_CANDIDATE_TAG"
      printf 'learning_rate=%s\n' "$ANCHORCAL_LEARNING_RATE"
      printf 'weight_decay=%s\n' "$ANCHORCAL_WEIGHT_DECAY"
    fi
  } > "$temporary"
  mv -- "$temporary" "$receipt"
  (
    cd -- "$receipt_root"
    sha256sum -- "$(basename -- "$receipt")" > "$(basename -- "$receipt").sha256"
  )
  ANCHORCAL_JOB_RECEIPT=$receipt
  ANCHORCAL_JOB_RECEIPT_SHA256=$(anchorcal_sha256 "$receipt")
  ANCHORCAL_JOB_RECEIPT_SERIES="$receipt_root/${stage}_job${SLURM_JOB_ID:-unknown}_attempt*.txt"
  export ANCHORCAL_JOB_RECEIPT ANCHORCAL_JOB_RECEIPT_SHA256 ANCHORCAL_JOB_RECEIPT_SERIES
  printf 'job_receipt=%s\n' "$receipt"
}

anchorcal_prepare_runtime() {
  local network_mode=$1
  local run_seed=$2
  local stage=$3
  local hardware=$4

  local required=(
    ANCHORCAL_REPO
    ANCHORCAL_EXPECTED_COMMIT
    ANCHORCAL_CONFIG
    ANCHORCAL_CONFIG_SHA256
    ANCHORCAL_DEBUG_CONFIG
    ANCHORCAL_DEBUG_CONFIG_SHA256
    ANCHORCAL_PATHS
    ANCHORCAL_PATHS_SHA256
    ANCHORCAL_PYTHON
    ANCHORCAL_PYTHON_SHA256
    ANCHORCAL_ENVIRONMENT_RECEIPT
    ANCHORCAL_ENVIRONMENT_RECEIPT_SHA256
    ANCHORCAL_PACKAGE_LOCK
    ANCHORCAL_PACKAGE_LOCK_SHA256
    ANCHORCAL_INPUT_RECEIPT
    ANCHORCAL_INPUT_RECEIPT_SHA256
    ANCHORCAL_OUTPUT_ROOT
  )
  local name
  for name in "${required[@]}"; do
    anchorcal_require_var "$name"
  done

  [[ "$ANCHORCAL_REPO" == "/home/ryreu/guided_cnn/BirdOnly" ]] || \
    anchorcal_die "unexpected repository root: ${ANCHORCAL_REPO}"
  [[ "$ANCHORCAL_OUTPUT_ROOT" == "/home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot" ]] || \
    anchorcal_die "unexpected campaign output root: ${ANCHORCAL_OUTPUT_ROOT}"
  [[ "$(uname -m)" == "aarch64" ]] || \
    anchorcal_die "TIGRIS jobs require aarch64, found $(uname -m)"
  cd -- "$ANCHORCAL_REPO"

  export PYTHONNOUSERSITE=1
  export PYTHONUNBUFFERED=1
  export PYTHONHASHSEED="$run_seed"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export HF_HOME=/home/ryreu/.cache/huggingface
  export TORCH_HOME=/home/ryreu/.cache/torch
  export PYTHONPATH="$ANCHORCAL_REPO/src"
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
  export TOKENIZERS_PARALLELISM=false

  case "$network_mode" in
    online)
      unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
      ;;
    offline)
      export HF_HUB_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1
      ;;
    *)
      anchorcal_die "network mode must be online or offline, found ${network_mode}"
      ;;
  esac

  anchorcal_verify_file "pilot config" "$ANCHORCAL_CONFIG" "$ANCHORCAL_CONFIG_SHA256"
  anchorcal_verify_file "debug config" "$ANCHORCAL_DEBUG_CONFIG" "$ANCHORCAL_DEBUG_CONFIG_SHA256"
  anchorcal_verify_file "frozen path config" "$ANCHORCAL_PATHS" "$ANCHORCAL_PATHS_SHA256"
  anchorcal_verify_file "Python executable" "$ANCHORCAL_PYTHON" "$ANCHORCAL_PYTHON_SHA256"
  anchorcal_verify_file "submission environment receipt" "$ANCHORCAL_ENVIRONMENT_RECEIPT" "$ANCHORCAL_ENVIRONMENT_RECEIPT_SHA256"
  anchorcal_verify_file "exact package lock" "$ANCHORCAL_PACKAGE_LOCK" "$ANCHORCAL_PACKAGE_LOCK_SHA256"
  anchorcal_verify_file "frozen-input receipt" "$ANCHORCAL_INPUT_RECEIPT" "$ANCHORCAL_INPUT_RECEIPT_SHA256"
  anchorcal_verify_package_lock

  if [[ "$hardware" == "gpu" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || anchorcal_die "nvidia-smi is unavailable"
    ANCHORCAL_GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)
    export ANCHORCAL_GPU_INFO
    printf '%s\n' "$ANCHORCAL_GPU_INFO"
  elif [[ "$hardware" != "cpu" ]]; then
    anchorcal_die "hardware mode must be gpu or cpu, found ${hardware}"
  else
    ANCHORCAL_GPU_INFO=none
    export ANCHORCAL_GPU_INFO
  fi
  ANCHORCAL_HARDWARE_MODE=$hardware
  export ANCHORCAL_HARDWARE_MODE

  echo "[$(date --iso-8601=seconds)] stage=${stage} host=$(hostname) job=${SLURM_JOB_ID:-unset}"
  echo "repo=${ANCHORCAL_REPO} commit=${ANCHORCAL_EXPECTED_COMMIT} network=${network_mode}"
  "$ANCHORCAL_PYTHON" --version
  anchorcal_write_job_receipt "$stage" "$network_mode"
}
