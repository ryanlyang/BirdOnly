#!/usr/bin/env bash
set -Eeuo pipefail

: "${SETV_ULA_REPO:?missing SETV_ULA_REPO}"
: "${SETV_ULA_SHADOW_DIR:?missing SETV_ULA_SHADOW_DIR}"
: "${SETV_ULA_SSL_DIR:?missing SETV_ULA_SSL_DIR}"

cd "$SETV_ULA_REPO"
python -u main_pretrain.py \
  --name setv_waterbirds95_official_mocov2plus \
  --method mocov2plus \
  --dataset waterbirds \
  --backbone resnet50 \
  --train_data_path "$SETV_ULA_SHADOW_DIR" \
  --valid_data_path "$SETV_ULA_SHADOW_DIR" \
  --test_data_path "$SETV_ULA_SHADOW_DIR" \
  --checkpoint_dir "$SETV_ULA_SSL_DIR" \
  --checkpoint_frequency 10 \
  --save_checkpoint \
  --auto_resume \
  --devices 0 \
  --accelerator gpu \
  --precision 16 \
  --num_workers 4 \
  --model_selection_metric "val_0/base/linear/acc1_0" \
  --model_selection_mode max \
  --select_best_model \
  --task 1 0 \
  --max_epochs 100 \
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
  --batch_size 256 \
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
  --queue_size 4096 \
  --temperature 0.1 \
  --base_tau_momentum 0.99 \
  --final_tau_momentum 0.999 \
  --momentum_classifier
