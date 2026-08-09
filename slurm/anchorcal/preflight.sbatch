#!/usr/bin/env bash
#SBATCH --job-name=ac_preflight
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

actual_commit=$(git -C "$ANCHORCAL_REPO" rev-parse HEAD) || {
  echo "AnchorCal preflight requires a usable Git checkout: $ANCHORCAL_REPO" >&2
  exit 2
}
[[ "$actual_commit" == "$ANCHORCAL_EXPECTED_COMMIT" ]] || {
  echo "AnchorCal commit changed: expected=$ANCHORCAL_EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
}
worktree_status=$(git -C "$ANCHORCAL_REPO" status --porcelain --untracked-files=normal)
[[ -z "$worktree_status" ]] || {
  echo "AnchorCal production checkout became dirty before preflight:" >&2
  printf '%s\n' "$worktree_status" >&2
  exit 2
}

# The commit/clean-tree guard above intentionally precedes project shell/Python code.
source "$ANCHORCAL_REPO/slurm/anchorcal/runtime_common.sh"
anchorcal_prepare_runtime online 1729 preflight gpu

srun --unbuffered "$ANCHORCAL_PYTHON" -u scripts/anchorcal/preflight.py \
  --config "$ANCHORCAL_CONFIG" \
  --paths "$ANCHORCAL_PATHS" \
  --allow-download \
  --require-gh200

preflight_files=(
  preflight/report.json
  preflight/resolved_config.yaml
  preflight/mask_manifest.json
  preflight/pretrained_manifest.json
  preflight/preprocessing_manifest.json
  environment/environment.json
  environment/package-lock.txt
  splits/manifest.json
  splits/waterbirds100_candidate_train.csv
  splits/waterbirds100_biased_val.csv
  splits/waterbirds100_expert_train.csv
  splits/waterbirds100_expert_calibration.csv
  splits/waterbirds100_oracle_val.csv
  splits/waterbirds100_test.csv
  preflight/geometry/expert_train_geometry.csv
  preflight/geometry/expert_calibration_geometry.csv
  preflight/geometry/biased_val_geometry.csv
  preflight/geometry/background_token_budget.json
  preflight/geometry/fixed_background_views.h5
  preflight/geometry/fixed_background_views.h5.manifest.json
  preflight/geometry/selector_eval_subset.csv
  preflight/geometry/donor_assignments.json
  preflight/geometry/manifest.json
)
for relative in "${preflight_files[@]}"; do
  [[ -f "$ANCHORCAL_OUTPUT_ROOT/$relative" ]] || \
    anchorcal_die "preflight did not create ${relative}"
done
(
  cd -- "$ANCHORCAL_OUTPUT_ROOT"
  checksum_temporary="preflight/.preflight_artifacts.${SLURM_JOB_ID}.${BASHPID}.tmp"
  sha256sum -- "${preflight_files[@]}" > "$checksum_temporary"
  mv -- "$checksum_temporary" preflight/preflight_artifacts.sha256
  sha256sum -c -- preflight/preflight_artifacts.sha256
)
