#!/usr/bin/env bash
#SBATCH --job-name=setv_set_capacity
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.err

set -Eeuo pipefail
: "${SETV_REPO:?missing SETV_REPO}"
: "${SETV_EXPECTED_COMMIT:?missing SETV_EXPECTED_COMMIT}"
: "${SETV_SET_SEED:?missing SETV_SET_SEED}"

ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
ROOT=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95
JSON_REPORT="${ROOT}/preflight/set_capacity_job${SLURM_JOB_ID}.json"
CSV_REPORT="${ROOT}/preflight/set_capacity_samples_job${SLURM_JOB_ID}.csv"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$SETV_REPO"

actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != "$SETV_EXPECTED_COMMIT" ]]; then
  echo "Capacity-audit provenance failure: submitted commit changed" >&2
  echo "expected=$SETV_EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Capacity-audit provenance failure: worktree became dirty" >&2
  git status --short >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"
echo "commit=$actual_commit"
python -c "import numpy,pandas,PIL,yaml; print(numpy.__version__,pandas.__version__,PIL.__version__,yaml.__version__)"
srun --unbuffered python -u scripts/audit_set_capacity.py \
  --config configs/expert_background_set.yaml \
  --seed "$SETV_SET_SEED" \
  --json-report "$JSON_REPORT" \
  --csv-report "$CSV_REPORT"

echo "Capacity census JSON: $JSON_REPORT"
echo "Capacity census CSV: $CSV_REPORT"
