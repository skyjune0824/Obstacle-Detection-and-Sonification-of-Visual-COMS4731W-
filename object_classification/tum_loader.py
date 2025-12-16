import os
import cv2
import numpy as np

class TUMDataLoader:
    """Load TUM RGB-D dataset format"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.rgb_dir = os.path.join(dataset_path, 'rgb')
        self.depth_dir = os.path.join(dataset_path, 'depth')
        
        # Load associations
        self.rgb_list = self._load_file_list('rgb.txt')
        self.depth_list = self._load_file_list('depth.txt')
        
        # Associate RGB and depth frames
        self.associations = self._associate_frames()
        
        print(f"Loaded TUM dataset: {len(self.associations)} frame pairs")
    
    def _load_file_list(self, filename):
        """Load timestamp and filename pairs from .txt file"""
        filepath = os.path.join(self.dataset_path, filename)
        data = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    timestamp = float(parts[0])
                    filename = parts[1]
                    data.append((timestamp, filename))
        
        return sorted(data, key=lambda x: x[0])
    
    def _associate_frames(self, max_time_diff=0.02):
        """Associate RGB and depth frames by timestamp"""
        associations = []
        
        for rgb_ts, rgb_file in self.rgb_list:
            # Find closest depth frame
            best_match = None
            best_diff = float('inf')
            
            for depth_ts, depth_file in self.depth_list:
                diff = abs(rgb_ts - depth_ts)
                if diff < best_diff and diff < max_time_diff:
                    best_diff = diff
                    best_match = (depth_ts, depth_file)
            
            if best_match:
                associations.append({
                    'rgb_timestamp': rgb_ts,
                    'rgb_file': rgb_file,
                    'depth_timestamp': best_match[0],
                    'depth_file': best_match[1]
                })
        
        return associations
    
    def get_frame_pair(self, index):
        """Get RGB and depth frame pair by index"""
        if index >= len(self.associations):
            return None, None
        
        assoc = self.associations[index]
        
        # Load RGB
        rgb_path = os.path.join(self.dataset_path, assoc['rgb_file'])
        rgb = cv2.imread(rgb_path)
        
        # Load depth (16-bit PNG, values in millimeters)
        depth_path = os.path.join(self.dataset_path, assoc['depth_file'])
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        # Convert depth from mm to meters
        depth = depth.astype(np.float32) / 5000.0  # TUM uses factor of 5000
        
        return rgb, depth
    
    def __len__(self):
        return len(self.associations)
