#!/usr/bin/env bash
#SBATCH --job-name=setv_exact_fusion
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.err

set -Eeuo pipefail
: "${SETV_REPO:?missing SETV_REPO}"
: "${SETV_OBJECT_SEED:?missing SETV_OBJECT_SEED}"
: "${SETV_EXACT_SEED:?missing SETV_EXACT_SEED}"
: "${SETV_EXACT_FUSION_SEED:?missing SETV_EXACT_FUSION_SEED}"

ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
ROOT=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95
OBJECT="${ROOT}/object_expert/seed_${SETV_OBJECT_SEED}"
EXACT="${ROOT}/background_exact/seed_${SETV_EXACT_SEED}"
FUSION="${ROOT}/fusion_exact/object_${SETV_OBJECT_SEED}_exact_${SETV_EXACT_SEED}_fusion_${SETV_EXACT_FUSION_SEED}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$SETV_REPO"
python -c "import numpy,pandas,scipy,sklearn; print(numpy.__version__,pandas.__version__,scipy.__version__,sklearn.__version__)"
python -u scripts/build_exact_fusions.py \
  --config configs/fusion_exact.yaml \
  --seed "$SETV_EXACT_FUSION_SEED" \
  --object-expert-dir "$OBJECT" \
  --exact-expert-dir "$EXACT"
python -u scripts/verify_exact_fusions.py --fusion-dir "$FUSION"
