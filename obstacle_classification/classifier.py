## After the training, the model will automatically be saved to the path: runs/detect/yolov8n_coco_exp_01/weights/best.pt


import cv2
import numpy as np
from ultralytics import YOLO
import torch

class ClassificationModule:
    """
    Object classification using your custom-trained YOLOv8 model
    """
    def __init__(self, model_path, conf_threshold=0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load your custom trained .pt file
        print(f"Loading model from {model_path} ...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf_threshold = conf_threshold

        # Map COCO classes to navigation classes
        self.coco_to_nav_mapping = {
            'person': 'pedestrian',
            'car': 'vehicle',
            'truck': 'vehicle',
            'bus': 'vehicle',
            'motorcycle': 'vehicle',
            'bicycle': 'vehicle'
        }
        self.target_classes = [0, 1, 2, 3, 5, 7]  # COCO indices for person, bicycle, car, motorcycle, bus, truck
        print(" Model loaded successfully!")
        print(f" Detecting classes: {list(self.coco_to_nav_mapping.keys())}")

    def detect_objects(self, rgb_frame, depth_map=None):
        results = self.model(rgb_frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.target_classes:
                    continue
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_name = self.model.names[cls_id]
                nav_class = self.coco_to_nav_mapping.get(cls_name, 'unknown')
                if nav_class == 'unknown':
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                avg_depth = None
                depth_confidence = 1.0
                if depth_map is not None:
                    avg_depth, depth_confidence = self._calculate_object_depth(
                        depth_map, x1, y1, x2, y2
                    )
                    if depth_confidence < 0.3:
                        continue
                detection = {
                    'class': nav_class,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'depth': avg_depth,
                    'depth_confidence': depth_confidence,
                    'original_class': cls_name
                }
                detections.append(detection)
        return detections

    def _calculate_object_depth(self, depth_map, x1, y1, x2, y2):
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        depth_roi = depth_map[y1:y2, x1:x2]
        valid_depths = depth_roi[depth_roi > 0]
        if len(valid_depths) == 0:
            return None, 0.0
        avg_depth = np.median(valid_depths)
        valid_ratio = len(valid_depths) / depth_roi.size
        return float(avg_depth), float(valid_ratio)

    def prioritize_detections(self, detections):
        def priority_score(det):
            class_priority = {'pedestrian': 10, 'vehicle': 9}
            class_weight = class_priority.get(det['class'], 0)
            depth_weight = 1.0 / (det['depth'] + 0.1) if det['depth'] else 0
            conf_weight = det['confidence']
            return class_weight * 10 + depth_weight * 5 + conf_weight
        return sorted(detections, key=priority_score, reverse=True)

    def visualize_detections(self, rgb_frame, detections):
        vis_frame = rgb_frame.copy()
        class_colors = {
            'pedestrian': (255, 0, 0),    # Blue
            'vehicle': (0, 255, 0)         # Green
        }
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls = det['class']
            conf = det['confidence']
            depth = det['depth']
            orig_cls = det['original_class']
            color = class_colors.get(cls, (255, 255, 255))
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{orig_cls}: {conf:.2f}"
            if depth:
                label += f" | {depth:.2f}m"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return vis_frame


def main():
    print("="*60)
    print("Visual Navigation System - Classification Module")
    print("Using YOUR Trained YOLOv8 Model")
    print("="*60)

    # Replace this path with the path to your trained .pt file
    classifier = ClassificationModule(
        model_path='runs/detect/yolov8n_coco_exp_01/weights/best.pt',
        conf_threshold=0.5
    )

    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or a filename for video
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    print("\n🎥 Starting detection...")
    print("Press 'q' to quit\n")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Use real depth map if available
        depth_map = None
        detections = classifier.detect_objects(frame, depth_map)
        detections = classifier.prioritize_detections(detections)
        if frame_count % 30 == 0:
            print(f"\n--- Frame {frame_count} ---")
            for i, det in enumerate(detections[:3]):
                depth_str = f"{det['depth']:.1f}m" if det['depth'] else "unknown distance"
                print(f"  {i+1}. {det['original_class']} ({det['class']}) at {depth_str} - conf: {det['confidence']:.2f}")
                if det['depth'] and det['depth'] < 2.0:
                    print(f"      ⚠️  ALERT: {det['class']} approaching!")
        vis_frame = classifier.visualize_detections(frame, detections)
        cv2.putText(vis_frame, f"Detections: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Object Detection - Custom YOLOv8 Model', vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("\n Detection complete!")

if __name__ == "__main__":
    main()
