#!/usr/bin/env bash
#SBATCH --job-name=ac_anchors
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
  echo "AnchorCal production checkout became dirty before anchor evaluation:" >&2
  printf '%s\n' "$worktree_status" >&2
  exit 2
}

source "$ANCHORCAL_REPO/slurm/anchorcal/runtime_common.sh"
anchorcal_prepare_runtime offline 7002 anchors gpu
(
  cd -- "$ANCHORCAL_OUTPUT_ROOT"
  sha256sum -c -- preflight/preflight_artifacts.sha256
)

shopt -s nullglob
existing_decisions=("$ANCHORCAL_OUTPUT_ROOT/receipt"/anchorcal_decision_*.json)
shopt -u nullglob
(( ${#existing_decisions[@]} == 0 )) || \
  anchorcal_die "refusing to create a second AnchorCal decision receipt"

srun --unbuffered "$ANCHORCAL_PYTHON" -u scripts/anchorcal/evaluate_anchors.py \
  --config "$ANCHORCAL_CONFIG" \
  --paths "$ANCHORCAL_PATHS"

decision_receipt=$(anchorcal_verify_decision_receipt)
echo "verified_anchorcal_decision_receipt=$decision_receipt"
