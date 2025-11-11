#! /bin/bash
python inference.py \
  --model_dir "models" \
  --model_name "camera_movement/epoch=9-step=1000" \
  --output_path "output" \
  --video_path  "test/camera_movement/0001.mp4" \
  --prompt="[clip1] is the original source video, and [clip2] is its corresponding scribble map. [clip3] is another, different source video. In [clip4], the scribble map transformation applied from [clip1] to [clip2] is similarly applied to [clip3]." \
  --width 832 \
  --height 480 \
  --n_clipA 17 \
  --n_clipB 17

