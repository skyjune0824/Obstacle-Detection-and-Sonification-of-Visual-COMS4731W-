import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque

class RGBDObjectClassifier:
    def __init__(self, yolo_model_path, depth_confidence_threshold=0.3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.yolo = YOLO(yolo_model_path)
        self.yolo.to(self.device)
        self.depth_confidence_threshold = depth_confidence_threshold
        
        # Enhanced class mapping for navigation context
        self.navigation_classes = {
            'person': {'priority': 10, 'safe_distance': 2.0, 'audio_cue': 'pedestrian'},
            'car': {'priority': 9, 'safe_distance': 3.0, 'audio_cue': 'vehicle'},
            'truck': {'priority': 9, 'safe_distance': 4.0, 'audio_cue': 'vehicle'},
            'bus': {'priority': 9, 'safe_distance': 4.0, 'audio_cue': 'vehicle'},
            'bicycle': {'priority': 8, 'safe_distance': 1.5, 'audio_cue': 'bicycle'},
            'motorcycle': {'priority': 8, 'safe_distance': 2.0, 'audio_cue': 'vehicle'},
            'chair': {'priority': 5, 'safe_distance': 1.0, 'audio_cue': 'furniture'},
            'bench': {'priority': 5, 'safe_distance': 1.0, 'audio_cue': 'furniture'},
        }
        
        # COCO class IDs for target classes
        self.target_coco_ids = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 7: 'truck', 56: 'chair', 57: 'couch', 60: 'dining table'
        }
    
    def classify_with_depth_validation(self, rgb_frame, depth_map):
        """Enhanced classification that validates detections with depth consistency"""
        # Run YOLO inference
        results = self.yolo(rgb_frame, conf=0.5, verbose=False)
        validated_detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                
                # Only process target classes
                if cls_id not in self.target_coco_ids:
                    continue
                    
                cls_name = self.target_coco_ids[cls_id]
                
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0])
                
                # Validate with depth information
                depth_validation = self._validate_with_depth(
                    depth_map, x1, y1, x2, y2, cls_name
                )
                
                if depth_validation['is_valid']:
                    detection = {
                        'class': cls_name,
                        'original_class': cls_name,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'depth': depth_validation['avg_depth'],
                        'depth_confidence': depth_validation['confidence'],
                        'navigation_class': self.navigation_classes[cls_name]['audio_cue'],
                        'safe_distance': self.navigation_classes[cls_name]['safe_distance'],
                        'priority': self.navigation_classes[cls_name]['priority'],
                        'is_urgent': depth_validation['avg_depth'] < self.navigation_classes[cls_name]['safe_distance']
                    }
                    validated_detections.append(detection)
        
        return validated_detections
    
    def _validate_with_depth(self, depth_map, x1, y1, x2, y2, class_name):
        """Validate detection using depth consistency checks"""
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return {'is_valid': False, 'avg_depth': None, 'confidence': 0.0}
        
        depth_roi = depth_map[y1:y2, x1:x2]
        valid_depths = depth_roi[depth_roi > 0]
        
        if len(valid_depths) == 0:
            return {'is_valid': False, 'avg_depth': None, 'confidence': 0.0}
        
        avg_depth = np.median(valid_depths)
        valid_ratio = len(valid_depths) / depth_roi.size
        
        # Class-specific depth validation
        is_valid = self._class_specific_validation(class_name, avg_depth, valid_ratio)
        
        return {
            'is_valid': is_valid and valid_ratio > self.depth_confidence_threshold,
            'avg_depth': float(avg_depth),
            'confidence': float(valid_ratio)
        }
    
    def _class_specific_validation(self, class_name, depth, valid_ratio):
        """Apply class-specific validation rules"""
        if class_name in ['person', 'bicycle']:
            return depth < 10.0 and depth > 0.5
        elif class_name in ['car', 'truck', 'bus']:
            return 1.0 < depth < 50.0
        else:
            return 0.3 < depth < 15.0

    def visualize_detections(self, rgb_frame, detections):
        """Visualize detections on RGB frame"""
        vis_frame = rgb_frame.copy()
        class_colors = {
            'pedestrian': (0, 0, 255),    # Red
            'vehicle': (0, 255, 0),       # Green
            'bicycle': (255, 255, 0),     # Cyan
            'furniture': (255, 165, 0)    # Orange
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            nav_class = det['navigation_class']
            conf = det['confidence']
            depth = det['depth']
            
            color = class_colors.get(nav_class, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class']}: {conf:.2f}"
            if depth:
                label += f" | {depth:.1f}m"
                if det['is_urgent']:
                    label += " ⚠️"
            
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame