import argparse
import torch
from diffsynth import ModelManager, save_video, VideoData
from diffsynth.pipelines.univid_wan_video import Univid_WanVideoPipeline
import random
import os
from PIL import Image
import numpy as np
import imageio


parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--model_dir",
    type=str
)
parser.add_argument(
    "--model_name",
    type=str
)
parser.add_argument(
    "--output_path",
    type=str,
)
parser.add_argument(
    "--prompt",
    type=str,
    default=None
)

parser.add_argument(
    "--model",
    type=str,
    default="/hdd/u202420081000004/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
)
parser.add_argument(
    "--t5_model",
    type=str,
    default="/hdd/u202420081000004/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
)
parser.add_argument(
    "--vae_model",
    type=str,
    default="/hdd/u202420081000004/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
)
parser.add_argument(
    "--n_clipA",
    type=int,
    default=17
)
parser.add_argument(
    "--n_clipB",
    type=int,
    default=17
)
parser.add_argument(
    "--video_path",
    type=str,
)

parser.add_argument(
    "--height",
    type=int,
    default=256
)

parser.add_argument(
    "--width",
    type=int,
    default=256
)
args = parser.parse_args()







model_name = args.model_name

model_dir = args.model_dir
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models([
    f"{args.model}",
     f"{args.t5_model}",
     f"{args.vae_model}",
])

model_manager.load_lora(f"{model_dir}/{model_name}.ckpt", lora_alpha=1.0)
model_manager.to('cuda')
pipe = Univid_WanVideoPipeline.from_model_manager(model_manager, device="cuda")

pipe.enable_vram_management(num_persistent_param_in_dit=None)


output_dir = f"{args.output_path}/{model_name}"

os.makedirs(output_dir, exist_ok=True)
video_name = os.path.basename(args.video_path)

reader = imageio.get_reader(args.video_path)
total_frames = reader.count_frames()
group_size = total_frames // 4
frames_total = []


for i in range(4):
    frames = []
    for frame_id in range(group_size):
        if frame_id == args.n_clipA and (i==0 or i ==1):
            break
        elif frame_id == args.n_clipB and (i==2 or i ==3):
            break
        frame = reader.get_data(i * group_size + frame_id)
        frame = Image.fromarray(frame)
        frames.append(frame)
    frames_total.append(frames)
reader.close()
group_size = [len(v) for v in frames_total]


for j in range(1):
    seed = random.randint(1, 9999)
    video = pipe(
        prompt=args.prompt,
        input_video=frames_total,
        num_inference_steps=50,
        height=args.height,
        width=args.width,
        num_frames=group_size,
        seed=seed, tiled=True
    )

    save_video(video, fr"{output_dir}/{video_name}", fps=8, quality=5)