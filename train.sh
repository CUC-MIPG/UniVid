#! /bin/bash


python examples/wanvideo/train_univid.py \
  --task train \
  --train_architecture lora \
  --dataset_path "data/train/camera_movement" \
  --output_path models \
  --exp_name camera_movement \
  --dit_path "Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors,Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth,Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --steps_per_epoch 200 \
  --prompt="[clip1] is the original source video, and [clip2] is the same clip with camera motion applied. [clip3] is a different source video. For [clip4], apply the exact same camera movement transformation from [clip1] to [clip2] on [clip3]." \
  --val_img_path="data/test/camera_movement/0001.mp4"\
  --max_epochs 20 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing\
  --width 832 \
  --height 480 \
  --n_clipA 17 \
  --n_clipB 17