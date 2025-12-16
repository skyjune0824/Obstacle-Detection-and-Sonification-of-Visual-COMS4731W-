import torch
import torch.nn as nn
from ultralytics import YOLO

class DualPathwayYOLO(nn.Module):
    def __init__(self, backbone_rgb, backbone_depth):
        super(DualPathwayYOLO, self).__init__()
        self.backbone_rgb = backbone_rgb
        self.backbone_depth = backbone_depth
        # Define fusion and prediction layers
        
    def forward(self, x):
        # x is [B, 4, H, W] - RGB + Depth concatenated
        rgb = x[:, :3, :, :]      # Extract RGB channels
        depth = x[:, 3:, :, :]    # Extract depth channel
        
        # Process through separate backbones
        features_rgb = self.backbone_rgb(rgb)
        features_depth = self.backbone_depth(depth)
        
        # Fuse features via concatenation
        fused_features = torch.cat((features_rgb, features_depth), dim=1)
        
        return fused_features


# Replace RealSense depth acquisition with your depth estimation
import cv2
import numpy as np

# Load your RGB frame
rgb_frame = cv2.imread('your_image.jpg')

# Load your monocular depth map (from Depth Anything V2)
depth_map = np.load('your_depth_map.npy')  # or load from your pipeline

# Normalize depth to match expected format
depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_normalized = depth_normalized.astype(np.uint8)

# Concatenate for 4-channel input
rgbd_input = np.concatenate([rgb_frame, depth_normalized[:,:,np.newaxis]], axis=2)
