#!/usr/bin/env bash
# Submit the complete locked AnchorCal Waterbirds100 TIGRIS campaign.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
EXPECTED_REPO=/home/ryreu/guided_cnn/BirdOnly
OUTPUT_ROOT=/home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot
CONFIG_SOURCE="$REPO/configs/anchorcal/pilot.yaml"
DEBUG_CONFIG="$REPO/configs/anchorcal/debug.yaml"
PATHS_SOURCE="$REPO/configs/anchorcal/paths.local.yaml"
DEFAULT_PYTHON=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200/bin/python
ANCHORCAL_PYTHON=${ANCHORCAL_PYTHON:-$DEFAULT_PYTHON}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"

die() {
  echo "AnchorCal submission refused: $*" >&2
  exit 2
}

sha256_of() {
  local digest ignored
  read -r digest ignored < <(sha256sum -- "$1")
  printf '%s\n' "$digest"
}

# Slurm gives SBATCH_* environment variables precedence over directives in the
# batch file. Refuse ambient overrides rather than silently changing a locked
# account, partition, GPU, resource, QOS, log path, or job name.
ambient_sbatch_variables=()
while IFS='=' read -r variable_name ignored_value; do
  if [[ "$variable_name" == SBATCH_* ]]; then
    ambient_sbatch_variables+=("$variable_name")
  fi
done < <(env)
if (( ${#ambient_sbatch_variables[@]} )); then
  printf 'Unset these ambient Slurm submission overrides first: %s\n' \
    "${ambient_sbatch_variables[*]}" >&2
  die "SBATCH_* environment variables would override the locked batch directives"
fi

[[ "$REPO" == "$EXPECTED_REPO" ]] || \
  die "run the committed launcher at ${EXPECTED_REPO}; found ${REPO}"
command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable; run this on tigris.rc.rit.edu"
command -v scancel >/dev/null 2>&1 || die "scancel is unavailable; partial submission cannot be rolled back"
command -v flock >/dev/null 2>&1 || die "flock is unavailable; concurrent submission cannot be prevented"
[[ "$ANCHORCAL_PYTHON" == /* ]] || die "ANCHORCAL_PYTHON must be an absolute path"
[[ -x "$ANCHORCAL_PYTHON" ]] || die "frozen interpreter is not executable: ${ANCHORCAL_PYTHON}"
[[ -f "$CONFIG_SOURCE" ]] || die "pilot config is missing: ${CONFIG_SOURCE}"
[[ -f "$DEBUG_CONFIG" ]] || die "debug config is missing: ${DEBUG_CONFIG}"
[[ -f "$PATHS_SOURCE" ]] || \
  die "create configs/anchorcal/paths.local.yaml from paths.local.example.yaml"

git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
  die "authoritative repository is not a Git checkout"
[[ "$(git -C "$REPO" rev-parse --show-toplevel)" == "$EXPECTED_REPO" ]] || \
  die "Git top level is not ${EXPECTED_REPO}"
worktree_status=$(git -C "$REPO" status --porcelain --untracked-files=normal)
if [[ -n "$worktree_status" ]]; then
  echo "$worktree_status" >&2
  die "production submission requires a clean Git worktree"
fi
ANCHORCAL_EXPECTED_COMMIT=$(git -C "$REPO" rev-parse HEAD)
cd -- "$REPO"

# Validate the local path overlay without importing any AnchorCal project code.
"$ANCHORCAL_PYTHON" -c 'import pathlib, sys, yaml
source = pathlib.Path(sys.argv[1])
payload = yaml.safe_load(source.read_text(encoding="utf-8"))
paths = payload.get("paths", payload) if isinstance(payload, dict) else None
required = (
    "repo_root", "waterbirds_root", "metadata_path",
    "vlm_mask_root",
    "hf_home", "output_root",
)
if not isinstance(paths, dict):
    raise SystemExit("paths.local.yaml must contain a paths mapping")
for key in required:
    value = paths.get(key)
    if not isinstance(value, str) or not value or value == "REQUIRED_ABSOLUTE_PATH":
        raise SystemExit(f"paths.{key} is unresolved")
    if not pathlib.Path(value).expanduser().is_absolute():
        raise SystemExit(f"paths.{key} is not absolute: {value}")
fixed = {
    "repo_root": "/home/ryreu/guided_cnn/BirdOnly",
    "waterbirds_root": "/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2",
    "metadata_path": "/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2/metadata.csv",
    "vlm_mask_root": "/home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds95_openclip_laion_dinovit/val/prediction_cmap",
    "output_root": "/home/ryreu/guided_cnn/BirdOnly/outputs/anchorcal/waterbirds100_pilot",
    "hf_home": "/home/ryreu/.cache/huggingface",
}
for key, expected in fixed.items():
    if str(pathlib.Path(paths[key]).resolve()) != expected:
        raise SystemExit(f"paths.{key} must resolve to {expected}, found {paths[key]}")
expected_metadata = pathlib.Path(paths["waterbirds_root"]).resolve() / "metadata.csv"
if pathlib.Path(paths["metadata_path"]).resolve() != expected_metadata:
    raise SystemExit("paths.metadata_path must be <waterbirds_root>/metadata.csv")
for key in ("repo_root", "waterbirds_root", "vlm_mask_root"):
    if not pathlib.Path(paths[key]).is_dir():
        raise SystemExit(f"paths.{key} is not an existing directory: {paths[key]}")
metadata_value = paths["metadata_path"]
if not pathlib.Path(metadata_value).is_file():
    raise SystemExit(f"paths.metadata_path is not an existing file: {metadata_value}")
' "$PATHS_SOURCE"

# Slurm opens stdout/stderr before the job body, so these directories must
# exist at submission time.
mkdir -p -- \
  "$OUTPUT_ROOT/run_logs" \
  "$OUTPUT_ROOT/manifests" \
  "$OUTPUT_ROOT/submission_receipts" \
  "$OUTPUT_ROOT/submission_receipts/jobs"

# Serialize launchers targeting the one locked output root, then repeat all
# collision checks while holding the lock.  The descriptor remains open until
# this process exits.
submission_lock="$OUTPUT_ROOT/submission_receipts/.submit_campaign.lock"
exec {submission_lock_fd}> "$submission_lock"
flock -n "$submission_lock_fd" || die "another submit_campaign.sh holds ${submission_lock}"

# A full submission is intentionally single-use. Candidate jobs themselves
# are restart-safe, but resubmitting the anchor stage would create a second
# immutable decision receipt and invalidate the campaign.
for production_path in debug branches anchors receipt candidates analysis; do
  if [[ -e "$OUTPUT_ROOT/$production_path" ]]; then
    die "existing campaign artifact prevents a new full graph: ${OUTPUT_ROOT}/${production_path}"
  fi
done
if ! prior_manifest=$(find "$OUTPUT_ROOT/manifests" -mindepth 1 -maxdepth 1 -type d -print -quit); then
  die "could not inspect prior campaign manifests"
fi
if [[ -n "$prior_manifest" ]]; then
  die "a prior frozen campaign manifest already exists under ${OUTPUT_ROOT}/manifests"
fi

active_names=ac_preflight,ac_debug,ac_foreground,ac_background,ac_anchors,ac_candidate,ac_cand_00,ac_cand_01,ac_cand_02,ac_cand_03,ac_cand_04,ac_cand_05,ac_final
if ! active_jobs=$(squeue -h -u "${USER:?USER is not set}" -n "$active_names"); then
  die "squeue failed; refusing to assume no colliding campaign exists"
fi
if [[ -n "$active_jobs" ]]; then
  printf '%s\n' "$active_jobs" >&2
  die "an AnchorCal job is already queued or running for ${USER}"
fi

campaign_id="$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
manifest_dir="$OUTPUT_ROOT/manifests/campaign_${campaign_id}"
mkdir -- "$manifest_dir"

CONFIG_FROZEN="$manifest_dir/pilot.yaml"
PATHS_FROZEN="$manifest_dir/paths.local.yaml"
PACKAGE_LOCK="$manifest_dir/package-lock.txt"
ENVIRONMENT_RECEIPT="$manifest_dir/environment.txt"
CANDIDATE_GRID="$manifest_dir/candidate-grid.tsv"
INPUT_RECEIPT="$manifest_dir/frozen-inputs.txt"

"$ANCHORCAL_PYTHON" -c 'import pathlib, sys, yaml
source, target, interpreter = map(pathlib.Path, sys.argv[1:])
payload = yaml.safe_load(source.read_text(encoding="utf-8"))
payload.setdefault("runtime", {})["python"] = str(interpreter)
target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
' "$CONFIG_SOURCE" "$CONFIG_FROZEN" "$ANCHORCAL_PYTHON"
cp -- "$PATHS_SOURCE" "$PATHS_FROZEN"

"$ANCHORCAL_PYTHON" -c 'from importlib import metadata
rows = {
    "{}=={}".format(dist.metadata["Name"], dist.version)
    for dist in metadata.distributions()
    if dist.metadata.get("Name")
}
print("\n".join(sorted(rows, key=str.casefold)))' > "$PACKAGE_LOCK"

python_sha256=$(sha256_of "$ANCHORCAL_PYTHON")
{
  printf 'schema_version=anchorcal-submission-environment-v1\n'
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'requested_python=%s\n' "$ANCHORCAL_PYTHON"
  printf 'resolved_python=%s\n' "$(realpath -- "$ANCHORCAL_PYTHON")"
  printf 'python_sha256=%s\n' "$python_sha256"
  "$ANCHORCAL_PYTHON" -c 'import platform, sys
print(f"python_version={platform.python_version()}")
print(f"python_implementation={platform.python_implementation()}")
print(f"submission_architecture={platform.machine()}")
print(f"submission_hostname={platform.node()}")
print(f"sys_executable={sys.executable}")'
} > "$ENVIRONMENT_RECEIPT"

{
  printf 'index\tlearning_rate\tweight_decay\n'
  printf '00\t1e-5\t0.01\n'
  printf '01\t1e-5\t0.05\n'
  printf '02\t3e-5\t0.01\n'
  printf '03\t3e-5\t0.05\n'
  printf '04\t1e-4\t0.01\n'
  printf '05\t1e-4\t0.05\n'
} > "$CANDIDATE_GRID"

config_sha256=$(sha256_of "$CONFIG_FROZEN")
debug_config_sha256=$(sha256_of "$DEBUG_CONFIG")
paths_sha256=$(sha256_of "$PATHS_FROZEN")
package_lock_sha256=$(sha256_of "$PACKAGE_LOCK")
environment_sha256=$(sha256_of "$ENVIRONMENT_RECEIPT")
candidate_grid_sha256=$(sha256_of "$CANDIDATE_GRID")
{
  printf 'schema_version=anchorcal-frozen-inputs-v1\n'
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'campaign_id=%s\n' "$campaign_id"
  printf 'expected_commit=%s\n' "$ANCHORCAL_EXPECTED_COMMIT"
  printf 'repo=%s\n' "$REPO"
  printf 'output_root=%s\n' "$OUTPUT_ROOT"
  printf 'pilot_config=%s\n' "$CONFIG_FROZEN"
  printf 'pilot_config_sha256=%s\n' "$config_sha256"
  printf 'pilot_source_config_sha256=%s\n' "$(sha256_of "$CONFIG_SOURCE")"
  printf 'debug_config=%s\n' "$DEBUG_CONFIG"
  printf 'debug_config_sha256=%s\n' "$debug_config_sha256"
  printf 'paths_config=%s\n' "$PATHS_FROZEN"
  printf 'paths_config_sha256=%s\n' "$paths_sha256"
  printf 'python=%s\n' "$ANCHORCAL_PYTHON"
  printf 'python_sha256=%s\n' "$python_sha256"
  printf 'environment_receipt=%s\n' "$ENVIRONMENT_RECEIPT"
  printf 'environment_receipt_sha256=%s\n' "$environment_sha256"
  printf 'package_lock=%s\n' "$PACKAGE_LOCK"
  printf 'package_lock_sha256=%s\n' "$package_lock_sha256"
  printf 'candidate_grid=%s\n' "$CANDIDATE_GRID"
  printf 'candidate_grid_sha256=%s\n' "$candidate_grid_sha256"
  printf 'implementation_plan_sha256=%s\n' "$(sha256_of "$REPO/AnchorCal_Waterbirds100_Pilot_Implementation_Plan.md")"
  printf 'decision_locks_sha256=%s\n' "$(sha256_of "$REPO/AnchorCal_Implementation_Decision_Locks_Answers.md")"
  printf 'pyproject_sha256=%s\n' "$(sha256_of "$REPO/pyproject.toml")"
  printf 'runtime_common_sha256=%s\n' "$(sha256_of "$REPO/slurm/anchorcal/runtime_common.sh")"
} > "$INPUT_RECEIPT"
input_receipt_sha256=$(sha256_of "$INPUT_RECEIPT")
printf '%s  %s\n' "$input_receipt_sha256" "$(basename -- "$INPUT_RECEIPT")" \
  > "${INPUT_RECEIPT}.sha256"
chmod 0444 -- "$manifest_dir"/*

export ANCHORCAL_REPO="$REPO"
export ANCHORCAL_EXPECTED_COMMIT
export ANCHORCAL_OUTPUT_ROOT="$OUTPUT_ROOT"
export ANCHORCAL_CONFIG="$CONFIG_FROZEN"
export ANCHORCAL_CONFIG_SHA256="$config_sha256"
export ANCHORCAL_DEBUG_CONFIG="$DEBUG_CONFIG"
export ANCHORCAL_DEBUG_CONFIG_SHA256="$debug_config_sha256"
export ANCHORCAL_PATHS="$PATHS_FROZEN"
export ANCHORCAL_PATHS_SHA256="$paths_sha256"
export ANCHORCAL_PYTHON
export ANCHORCAL_PYTHON_SHA256="$python_sha256"
export ANCHORCAL_ENVIRONMENT_RECEIPT="$ENVIRONMENT_RECEIPT"
export ANCHORCAL_ENVIRONMENT_RECEIPT_SHA256="$environment_sha256"
export ANCHORCAL_PACKAGE_LOCK="$PACKAGE_LOCK"
export ANCHORCAL_PACKAGE_LOCK_SHA256="$package_lock_sha256"
export ANCHORCAL_INPUT_RECEIPT="$INPUT_RECEIPT"
export ANCHORCAL_INPUT_RECEIPT_SHA256="$input_receipt_sha256"
unset ANCHORCAL_LEARNING_RATE ANCHORCAL_WEIGHT_DECAY ANCHORCAL_CANDIDATE_TAG

submitted_jobs=()
LAST_JOB_ID=
submit_job() {
  local raw
  if ! raw=$(sbatch --parsable --export=ALL "$@"); then
    echo "sbatch failed for: $*" >&2
    return 1
  fi
  LAST_JOB_ID=${raw%%;*}
  if [[ ! "$LAST_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Could not parse sbatch job ID from: ${raw}" >&2
    return 2
  fi
  submitted_jobs+=("$LAST_JOB_ID")
}

write_partial_receipt() {
  local status=$1
  local partial="$OUTPUT_ROOT/submission_receipts/campaign_${campaign_id}_${status}.txt"
  {
    printf 'schema_version=anchorcal-tigris-submission-v1\n'
    printf 'status=%s\n' "$status"
    printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'campaign_id=%s\n' "$campaign_id"
    printf 'expected_commit=%s\n' "$ANCHORCAL_EXPECTED_COMMIT"
    printf 'frozen_input_receipt=%s\n' "$INPUT_RECEIPT"
    printf 'frozen_input_receipt_sha256=%s\n' "$input_receipt_sha256"
    printf 'submitted_job_ids=%s\n' "${submitted_jobs[*]:-none}"
    printf 'rollback_status=%s\n' "${ROLLBACK_STATUS:-not_attempted}"
  } > "$partial"
  local digest
  digest=$(sha256_of "$partial")
  printf '%s  %s\n' "$digest" "$(basename -- "$partial")" > "${partial}.sha256"
}
handle_submission_error() {
  local status=$?
  trap - ERR
  ROLLBACK_STATUS=not_needed
  if (( ${#submitted_jobs[@]} )); then
    if scancel "${submitted_jobs[@]}"; then
      ROLLBACK_STATUS=scancel_requested
    else
      ROLLBACK_STATUS=scancel_failed
    fi
  fi
  write_partial_receipt submission_interrupted
  echo "Submission stopped after jobs: ${submitted_jobs[*]:-none}; rollback=${ROLLBACK_STATUS}" >&2
  exit "$status"
}
trap handle_submission_error ERR

submit_job "$REPO/slurm/anchorcal/preflight.sbatch"
preflight_id=$LAST_JOB_ID
submit_job --dependency="afterok:${preflight_id}" "$REPO/slurm/anchorcal/debug.sbatch"
debug_id=$LAST_JOB_ID
submit_job --dependency="afterok:${debug_id}" "$REPO/slurm/anchorcal/foreground.sbatch"
foreground_id=$LAST_JOB_ID
submit_job --dependency="afterok:${debug_id}" "$REPO/slurm/anchorcal/background.sbatch"
background_id=$LAST_JOB_ID
submit_job --dependency="afterok:${foreground_id}:${background_id}" "$REPO/slurm/anchorcal/anchors.sbatch"
anchors_id=$LAST_JOB_ID

learning_rates=(1e-5 1e-5 3e-5 3e-5 1e-4 1e-4)
weight_decays=(0.01 0.05 0.01 0.05 0.01 0.05)
candidate_ids=()
for index in "${!learning_rates[@]}"; do
  printf -v candidate_index '%02d' "$index"
  export ANCHORCAL_LEARNING_RATE=${learning_rates[$index]}
  export ANCHORCAL_WEIGHT_DECAY=${weight_decays[$index]}
  export ANCHORCAL_CANDIDATE_TAG="c${candidate_index}"
  submit_job \
    --job-name="ac_cand_${candidate_index}" \
    --dependency="afterok:${anchors_id}" \
    "$REPO/slurm/anchorcal/candidate.sbatch"
  candidate_ids+=("$LAST_JOB_ID")
done

candidate_dependency=$(IFS=:; printf '%s' "${candidate_ids[*]}")
unset ANCHORCAL_LEARNING_RATE ANCHORCAL_WEIGHT_DECAY ANCHORCAL_CANDIDATE_TAG
submit_job --dependency="afterok:${candidate_dependency}" "$REPO/slurm/anchorcal/final.sbatch"
final_id=$LAST_JOB_ID

submission_receipt="$OUTPUT_ROOT/submission_receipts/campaign_${campaign_id}.txt"
{
  printf 'schema_version=anchorcal-tigris-submission-v1\n'
  printf 'status=submitted\n'
  printf 'submitted_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'campaign_id=%s\n' "$campaign_id"
  printf 'expected_commit=%s\n' "$ANCHORCAL_EXPECTED_COMMIT"
  printf 'frozen_input_receipt=%s\n' "$INPUT_RECEIPT"
  printf 'frozen_input_receipt_sha256=%s\n' "$input_receipt_sha256"
  printf 'preflight_job_id=%s\n' "$preflight_id"
  printf 'debug_job_id=%s\n' "$debug_id"
  printf 'foreground_job_id=%s\n' "$foreground_id"
  printf 'background_job_id=%s\n' "$background_id"
  printf 'anchor_job_id=%s\n' "$anchors_id"
  for index in "${!candidate_ids[@]}"; do
    printf 'candidate_%02d_job_id=%s\n' "$index" "${candidate_ids[$index]}"
    printf 'candidate_%02d_learning_rate=%s\n' "$index" "${learning_rates[$index]}"
    printf 'candidate_%02d_weight_decay=%s\n' "$index" "${weight_decays[$index]}"
  done
  printf 'final_job_id=%s\n' "$final_id"
  printf 'dependency_graph=%s->%s->(%s,%s)->%s->(%s)->%s\n' \
    "$preflight_id" "$debug_id" "$foreground_id" "$background_id" \
    "$anchors_id" "${candidate_ids[*]}" "$final_id"
} > "$submission_receipt"
submission_sha256=$(sha256_of "$submission_receipt")
printf '%s  %s\n' "$submission_sha256" "$(basename -- "$submission_receipt")" \
  > "${submission_receipt}.sha256"
chmod 0444 -- "$submission_receipt" "${submission_receipt}.sha256"
trap - ERR

echo "Submitted AnchorCal campaign ${campaign_id} at commit ${ANCHORCAL_EXPECTED_COMMIT}"
echo "preflight=${preflight_id} debug=${debug_id} foreground=${foreground_id} background=${background_id} anchors=${anchors_id}"
echo "candidates=${candidate_ids[*]} final=${final_id}"
echo "receipt=${submission_receipt}"
