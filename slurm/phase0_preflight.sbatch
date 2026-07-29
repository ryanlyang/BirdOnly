#!/usr/bin/env bash
#SBATCH --job-name=setv_wb95_phase0
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.err

set -Eeuo pipefail

: "${SETV_REPO:?submit_phase0.sh must export SETV_REPO}"
: "${SETV_EXPECTED_COMMIT:?submit_phase0.sh must export SETV_EXPECTED_COMMIT}"

ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
CONFIG="${SETV_REPO}/configs/data_waterbirds95.yaml"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$SETV_REPO"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Phase 0 runtime provenance failure: repository is not a Git checkout" >&2
  exit 2
fi
actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != "$SETV_EXPECTED_COMMIT" ]]; then
  echo "Phase 0 runtime provenance failure: submitted commit changed" >&2
  echo "expected=$SETV_EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Phase 0 runtime provenance failure: worktree became dirty" >&2
  git status --short >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"
echo "repo=$SETV_REPO"
echo "commit=$actual_commit"
echo "architecture=$(uname -m)"
python --version
python -c "import numpy, pandas, PIL, yaml; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'Pillow', PIL.__version__, 'PyYAML', yaml.__version__)"
df -h /home/ryreu

test -f "$CONFIG"
test -d /home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2
test -d /home/ryreu/guided_cnn/Food101/LearningToLook/code/WeCLIPPlus/results_waterbirds95_openclip_laion_dinovit/val/prediction_cmap

PYTHONPATH=src python -m unittest discover -s tests -v
python -u scripts/build_phase0.py --config "$CONFIG"
python -u scripts/verify_phase0.py \
  --phase0-dir /home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/phase0 \
  --allow-pending-visual-review
