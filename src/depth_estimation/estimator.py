# Things to take into account for the model.
# Average walking speed: 1.10 m/s - 1.65 m/s: Avg: 1.4 m/s
# Starting Intuition: 1 frame per second (1 frame every 1.4 meters).
# Need a model capable of computing one depth map per second with negligible latency.
# https://huggingface.co/apple/coreml-depth-anything-v2-small - Depth Anything V2 optimized for mobile development
# iPhone 12 Pro Max - Inference time = 31.10 MS -> Assuming inference grows at a scale much larger than the other aspects of our computations,
# We could in theory take up to 30 images per second.
# Because Depth Anything V2 is supported by CoreML and we are considering a mobile implementation, we will use this model.

# Path Import for Depth Anything Model
from pathlib import Path
import sys

current_file = Path(__file__)
project_root = current_file.parent.parent.parent
da_path = project_root / "models" / "Depth-Anything-V2"
sys.path.insert(0, str(da_path))

print(da_path / "checkpoints" / "depth_anything_v2_")

import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])

WEIGHT_PATH = da_path / "checkpoints" / f"depth_anything_v2_{encoder}.pth"

model.load_state_dict(torch.load(str(WEIGHT_PATH), map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('your/image/path')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy