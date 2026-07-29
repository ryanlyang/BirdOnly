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

ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
CONFIG="${SETV_REPO}/configs/data_waterbirds95.yaml"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$SETV_REPO"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"
echo "repo=$SETV_REPO"
echo "commit=$(git rev-parse HEAD 2>/dev/null || printf UNKNOWN)"
echo "architecture=$(uname -m)"
git status --short 2>/dev/null || true
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
