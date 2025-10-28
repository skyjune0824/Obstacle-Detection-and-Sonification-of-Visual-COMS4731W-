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
