#!/usr/bin/env bash
#SBATCH --job-name=ac_debug
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
  echo "AnchorCal checkout became dirty before debug:" >&2
  printf '%s\n' "$worktree_status" >&2
  exit 2
}

source "$ANCHORCAL_REPO/slurm/anchorcal/runtime_common.sh"
anchorcal_prepare_runtime offline 9001 debug gpu
(
  cd -- "$ANCHORCAL_OUTPUT_ROOT"
  sha256sum -c -- preflight/preflight_artifacts.sha256
)

srun --unbuffered "$ANCHORCAL_PYTHON" -u scripts/anchorcal/run_debug.py \
  --config "$ANCHORCAL_CONFIG" \
  --paths "$ANCHORCAL_PATHS" \
  --skip-preflight

[[ -f "$ANCHORCAL_OUTPUT_ROOT/debug/analysis/summary.json" ]] || \
  anchorcal_die "standalone debug did not reach the final analysis artifact"
