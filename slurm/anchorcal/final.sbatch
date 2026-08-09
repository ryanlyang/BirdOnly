#!/usr/bin/env bash
#SBATCH --job-name=ac_final
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot/run_logs/%x_%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
: "${ANCHORCAL_REPO:?submit_campaign.sh must export ANCHORCAL_REPO}"
: "${ANCHORCAL_EXPECTED_COMMIT:?submit_campaign.sh must export ANCHORCAL_EXPECTED_COMMIT}"
actual_commit=$(git -C "$ANCHORCAL_REPO" rev-parse HEAD) || exit 2
[[ "$actual_commit" == "$ANCHORCAL_EXPECTED_COMMIT" ]] || {
  echo "AnchorCal commit changed: expected=$ANCHORCAL_EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
}
worktree_status=$(git -C "$ANCHORCAL_REPO" status --porcelain --untracked-files=normal)
[[ -z "$worktree_status" ]] || {
  echo "AnchorCal production checkout became dirty before final analysis:" >&2
  printf '%s\n' "$worktree_status" >&2
  exit 2
}

source "$ANCHORCAL_REPO/slurm/anchorcal/runtime_common.sh"
anchorcal_prepare_runtime offline 7003 final_analysis cpu
(
  cd -- "$ANCHORCAL_OUTPUT_ROOT"
  sha256sum -c -- preflight/preflight_artifacts.sha256
)
decision_receipt=$(anchorcal_verify_decision_receipt)
echo "verified_anchorcal_decision_receipt=$decision_receipt"
shopt -s nullglob
existing_selections=("$ANCHORCAL_OUTPUT_ROOT/receipt"/candidate_selection_*.json)
shopt -u nullglob
(( ${#existing_selections[@]} <= 1 )) || \
  anchorcal_die "multiple candidate-selection receipts make final analysis ambiguous"
if (( ${#existing_selections[@]} == 1 )); then
  [[ -f "${existing_selections[0]}.sha256" ]] || \
    anchorcal_die "existing candidate-selection receipt lacks its hash sidecar"
  echo "resuming reporting-only analysis from ${existing_selections[0]}"
fi

srun --unbuffered "$ANCHORCAL_PYTHON" -u scripts/anchorcal/final_analysis.py \
  --config "$ANCHORCAL_CONFIG" \
  --paths "$ANCHORCAL_PATHS"

[[ -f "$ANCHORCAL_OUTPUT_ROOT/analysis/summary.json" ]] || \
  anchorcal_die "final joined analysis artifact is missing"
