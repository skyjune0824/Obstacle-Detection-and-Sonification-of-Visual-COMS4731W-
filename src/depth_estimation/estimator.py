# Things to take into account for the model.
# Average walking speed: 1.10 m/s - 1.65 m/s: Avg: 1.4 m/s
# Starting Intuition: 1 frame per second (1 frame every 1.4 meters).
# Need a model capable of computing one depth map per second with negligible latency.
# https://huggingface.co/apple/coreml-depth-anything-v2-small - Depth Anything V2 optimized for mobile development
# iPhone 12 Pro Max - Inference time = 31.10 MS -> Assuming inference grows at a scale much larger than the other aspects of our computations,
# We could in theory take up to 30 images per second.
# Because Depth Anything V2 is supported by CoreML and we are considering a mobile implementation, we will use this model.

from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

img = Image.open("test_image.jpg")
out = pipe(img)
depth_map = np.array(out["depth"])

plt.imshow(depth_map, cmap="magma")
plt.colorbar()
plt.show()

# Output to file
depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_img = Image.fromarray((depth_norm * 255).astype(np.uint8))
depth_img.save("depth_map.png")

# (Optional) save raw float data
np.save("depth_map.npy", depth_map)
