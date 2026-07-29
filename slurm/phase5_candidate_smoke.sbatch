#!/usr/bin/env bash
#SBATCH --job-name=setv_candidate_smoke
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.err

set -Eeuo pipefail
: "${SETV_REPO:?missing SETV_REPO}"
: "${SETV_SMOKE_CANDIDATE_SEED:?missing SETV_SMOKE_CANDIDATE_SEED}"
: "${SETV_EXACT_FUSION_DIR:?missing SETV_EXACT_FUSION_DIR}"
: "${SETV_SANITIZED_FUSION_DIR:?missing SETV_SANITIZED_FUSION_DIR}"
: "${SETV_SET_FUSION_DIR:?missing SETV_SET_FUSION_DIR}"

ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
ROOT=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95
REPORT="${ROOT}/preflight/candidate_smoke_seed${SETV_SMOKE_CANDIDATE_SEED}_job${SLURM_JOB_ID}.json"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$SETV_REPO"
echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID} commit=$(git rev-parse HEAD)"
python -c "import torch,torchvision,timm; print(torch.__version__,torchvision.__version__,timm.__version__,torch.version.cuda)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
PYTHONPATH=src python -m unittest discover -s tests -v
srun --unbuffered python -u scripts/smoke_candidate.py \
  --config configs/candidate_erm.yaml \
  --seed "$SETV_SMOKE_CANDIDATE_SEED" \
  --exact-fusion-dir "$SETV_EXACT_FUSION_DIR" \
  --sanitized-fusion-dir "$SETV_SANITIZED_FUSION_DIR" \
  --set-fusion-dir "$SETV_SET_FUSION_DIR" \
  --report "$REPORT"
