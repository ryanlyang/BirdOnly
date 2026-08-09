#!/usr/bin/env bash
#SBATCH --job-name=ac_candidate
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot/run_logs/%x_%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
: "${ANCHORCAL_REPO:?submit_campaign.sh must export ANCHORCAL_REPO}"
: "${ANCHORCAL_EXPECTED_COMMIT:?submit_campaign.sh must export ANCHORCAL_EXPECTED_COMMIT}"
: "${ANCHORCAL_LEARNING_RATE:?submit_campaign.sh must export ANCHORCAL_LEARNING_RATE}"
: "${ANCHORCAL_WEIGHT_DECAY:?submit_campaign.sh must export ANCHORCAL_WEIGHT_DECAY}"
: "${ANCHORCAL_CANDIDATE_TAG:?submit_campaign.sh must export ANCHORCAL_CANDIDATE_TAG}"
actual_commit=$(git -C "$ANCHORCAL_REPO" rev-parse HEAD) || exit 2
[[ "$actual_commit" == "$ANCHORCAL_EXPECTED_COMMIT" ]] || {
  echo "AnchorCal commit changed: expected=$ANCHORCAL_EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
}
worktree_status=$(git -C "$ANCHORCAL_REPO" status --porcelain --untracked-files=normal)
[[ -z "$worktree_status" ]] || {
  echo "AnchorCal production checkout became dirty before candidate training:" >&2
  printf '%s\n' "$worktree_status" >&2
  exit 2
}

source "$ANCHORCAL_REPO/slurm/anchorcal/runtime_common.sh"
anchorcal_prepare_runtime offline 1234 "candidate_${ANCHORCAL_CANDIDATE_TAG}" gpu
(
  cd -- "$ANCHORCAL_OUTPUT_ROOT"
  sha256sum -c -- preflight/preflight_artifacts.sha256
)
decision_receipt=$(anchorcal_verify_decision_receipt)
echo "verified_anchorcal_decision_receipt=$decision_receipt"
echo "candidate_learning_rate=$ANCHORCAL_LEARNING_RATE candidate_weight_decay=$ANCHORCAL_WEIGHT_DECAY"

srun --unbuffered "$ANCHORCAL_PYTHON" -u scripts/anchorcal/train_candidate.py \
  --learning-rate "$ANCHORCAL_LEARNING_RATE" \
  --weight-decay "$ANCHORCAL_WEIGHT_DECAY" \
  --config "$ANCHORCAL_CONFIG" \
  --paths "$ANCHORCAL_PATHS"
