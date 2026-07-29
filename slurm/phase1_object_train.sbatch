#!/usr/bin/env bash
#SBATCH --job-name=setv_obj_train
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/run_logs/%x_%j.err

set -Eeuo pipefail

: "${SETV_REPO:?submit_phase1_object.sh must export SETV_REPO}"
: "${SETV_OBJECT_SEED:?submit_phase1_object.sh must export SETV_OBJECT_SEED}"

ENV=/home/ryreu/miniforge3-aarch64/envs/fcv_gh200
CONFIG="${SETV_REPO}/configs/expert_object_green.yaml"
OUTPUT_DIR="/home/ryreu/guided_cnn/logsWaterbird/setv_waterbirds95/object_expert/seed_${SETV_OBJECT_SEED}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/home/ryreu/.cache/huggingface
export TORCH_HOME=/home/ryreu/.cache/torch
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$SETV_REPO"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"
echo "commit=$(git rev-parse HEAD)"
echo "seed=$SETV_OBJECT_SEED"
python --version
python -c "import torch, torchvision, timm; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'timm', timm.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --unbuffered python -u scripts/train_object_expert.py \
  --config "$CONFIG" \
  --seed "$SETV_OBJECT_SEED"
python -u scripts/verify_object_expert.py --output-dir "$OUTPUT_DIR"

