#!/usr/bin/env bash
set -Eeuo pipefail

: "${SETV_ULA_REPO:?missing SETV_ULA_REPO}"
: "${SETV_ULA_SHADOW_DIR:?missing SETV_ULA_SHADOW_DIR}"
: "${SETV_ULA_SSL_DIR:?missing SETV_ULA_SSL_DIR}"
: "${SETV_ULA_SEED:?missing SETV_ULA_SEED}"

SETV_ULA_MAX_EPOCHS=${SETV_ULA_MAX_EPOCHS:-100}
SETV_ULA_BATCH_SIZE=${SETV_ULA_BATCH_SIZE:-256}
SETV_ULA_NUM_WORKERS=${SETV_ULA_NUM_WORKERS:-4}
SETV_ULA_QUEUE_SIZE=${SETV_ULA_QUEUE_SIZE:-4096}
SETV_ULA_PRECISION=${SETV_ULA_PRECISION:-16}
SETV_ULA_CHECKPOINT_FREQUENCY=${SETV_ULA_CHECKPOINT_FREQUENCY:-10}
SETV_ULA_RUN_NAME=${SETV_ULA_RUN_NAME:-setv_waterbirds95_official_mocov2plus}
for value in \
  "$SETV_ULA_MAX_EPOCHS" \
  "$SETV_ULA_BATCH_SIZE" \
  "$SETV_ULA_NUM_WORKERS" \
  "$SETV_ULA_QUEUE_SIZE" \
  "$SETV_ULA_CHECKPOINT_FREQUENCY"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "Official uLA operational overrides must be nonnegative integers" >&2
    exit 2
  fi
done
if (( SETV_ULA_MAX_EPOCHS < 1 || SETV_ULA_BATCH_SIZE < 1 || SETV_ULA_QUEUE_SIZE < 1 )); then
  echo "Official uLA epochs, batch size, and queue size must be positive" >&2
  exit 2
fi
if (( SETV_ULA_QUEUE_SIZE % SETV_ULA_BATCH_SIZE != 0 )); then
  echo "Official uLA queue size must be divisible by batch size" >&2
  exit 2
fi

cd "$SETV_ULA_REPO"
python -u main_pretrain.py \
  --name "$SETV_ULA_RUN_NAME" \
  --seed "$SETV_ULA_SEED" \
  --method mocov2plus \
  --dataset waterbirds \
  --backbone resnet50 \
  --train_data_path "$SETV_ULA_SHADOW_DIR" \
  --valid_data_path "$SETV_ULA_SHADOW_DIR" \
  --test_data_path "$SETV_ULA_SHADOW_DIR" \
  --checkpoint_dir "$SETV_ULA_SSL_DIR" \
  --checkpoint_frequency "$SETV_ULA_CHECKPOINT_FREQUENCY" \
  --save_checkpoint \
  --auto_resume \
  --devices 0 \
  --accelerator gpu \
  --precision "$SETV_ULA_PRECISION" \
  --num_workers "$SETV_ULA_NUM_WORKERS" \
  --model_selection_metric "val_0/base/linear/acc1_0" \
  --model_selection_mode max \
  --select_best_model \
  --task 1 0 \
  --max_epochs "$SETV_ULA_MAX_EPOCHS" \
  --optimizer lars \
  --scheduler warmup \
  --scheduler_interval step \
  --warmup_epochs 5 \
  --warmup_start_lr 0.00001 \
  --eta_lars 0.002 \
  --exclude_bias_n_norm \
  --grad_clip_lars \
  --lr 0.3 \
  --weight_decay 0.00003 \
  --classifier_lr 0.1 \
  --classifier_wd 0.0 \
  --batch_size "$SETV_ULA_BATCH_SIZE" \
  --num_crops_per_aug 2 \
  --augment strong \
  --min_scale 0.5 \
  --color_jitter_prob 0.8 \
  --brightness 0.4 \
  --contrast 0.4 \
  --saturation 0.4 \
  --hue 0.1 \
  --gray_scale_prob 0.2 \
  --gaussian_prob 0.5 \
  --solarization_prob 0.0 \
  --equalization_prob 0.0 \
  --horizontal_flip_prob 0.5 \
  --proj_output_dim 256 \
  --proj_hidden_dim 2048 \
  --queue_size "$SETV_ULA_QUEUE_SIZE" \
  --temperature 0.1 \
  --base_tau_momentum 0.99 \
  --final_tau_momentum 0.999 \
  --momentum_classifier
