#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
EXPECTED_REPO=/home/ryreu/guided_cnn/BirdOnly
ANCHORCAL_PYTHON=${ANCHORCAL_PYTHON:-/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python}

while IFS='=' read -r variable_name ignored_value; do
  if [[ "$variable_name" == SBATCH_* ]]; then
    echo "unset ambient Slurm override before diagnostic: $variable_name" >&2
    exit 2
  fi
done < <(env)

[[ "$REPO" == "$EXPECTED_REPO" ]] || {
  echo "run this from the authoritative TIGRIS checkout" >&2
  exit 2
}
[[ -x "$ANCHORCAL_PYTHON" ]] || {
  echo "AnchorCal interpreter is unavailable: $ANCHORCAL_PYTHON" >&2
  exit 2
}
[[ -f "$REPO/outputs/anchorcal/waterbirds100_pilot/branches/background/epoch_final.pt" ]] || {
  echo "the completed production background checkpoint is unavailable" >&2
  exit 2
}
status=$(git -C "$REPO" status --porcelain --untracked-files=normal)
[[ -z "$status" ]] || {
  printf '%s\n' "$status" >&2
  echo "diagnostic submission requires a clean checkout" >&2
  exit 2
}
mkdir -p \
  "$REPO/outputs/anchorcal/random_token_diagnostics/run_logs" \
  "$REPO/outputs/anchorcal/random_token_diagnostics/reports"
commit=$(git -C "$REPO" rev-parse HEAD)
job_id=$(sbatch --parsable \
  --export="ALL,ANCHORCAL_DIAGNOSTIC_COMMIT=$commit,ANCHORCAL_PYTHON=$ANCHORCAL_PYTHON" \
  "$REPO/slurm/anchorcal/random_token_diagnostic.sbatch")
printf 'Submitted random-token diagnostic job %s at commit %s\n' "$job_id" "$commit"
printf 'Report: %s/outputs/anchorcal/random_token_diagnostics/reports/random_token_job%s.json\n' \
  "$REPO" "$job_id"
