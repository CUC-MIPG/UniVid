#! /bin/bash

python examples/wanvideo/train_univid.py \
  --task data_process \
  --dataset_path  "data/train/camera_movement" \
  --text_encoder_path "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --tiled \
  --width 832 \
  --height 480 \
  --num_frames 180 \
  --n_clipA 17  \
  --n_clipB 17