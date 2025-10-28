# Things to take into account for the model.
# Average walking speed: 1.10 m/s - 1.65 m/s: Avg: 1.4 m/s
# Starting Intuition: 1 frame per second (1 frame every 1.4 meters).
# Need a model capable of computing one depth map per second with negligible latency.
# https://huggingface.co/apple/coreml-depth-anything-v2-small - Depth Anything V2 optimized for mobile development
# iPhone 12 Pro Max - Inference time = 31.10 MS -> Assuming inference grows at a scale much larger than the other aspects of our computations,
# We could in theory take up to 30 images per second.
# Because Depth Anything V2 is supported by CoreML and we are considering a mobile implementation, we will use this model.

import coremltools as ct
import numpy as np
from PIL import Image

# Load model
model = ct.models.MLModel("DepthAnythingV2SmallF16.mlpackage")

# Pre-process image to match model spec
img = Image.open("src/depth_estimation/test_image.jpg").convert("RGB")
img = img.resize((518, 518))  # or whatever size the model expects
img_np = np.array(img).astype(np.float32) / 255.0
# Possibly transpose to CHW or whatever the model expects
img_np = np.transpose(img_np, (2, 0, 1))[None, ...]

# Run prediction
out = model.predict({"image": img_np})  # key = input name
depth_map = out["depth"]  # key = output name