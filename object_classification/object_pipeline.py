import cv2
import numpy as np
from rgbd_classifier import RGBDObjectClassifier
from object_tracker import ObjectTracker
from prioritizer import ContextAwarePrioritizer
from audio_mapper import ObjectAwareAudioMapper

class ObjectClassificationPipeline:
    def __init__(self, yolo_model_path):
        self.classifier = RGBDObjectClassifier(yolo_model_path)
        self.tracker = ObjectTracker()
        self.prioritizer = ContextAwarePrioritizer()
        self.audio_mapper = ObjectAwareAudioMapper()
        self.frame_count = 0
    
    def process_frame(self, rgb_frame, depth_map):
        """Complete object classification pipeline"""
        self.frame_count += 1
        
        # Step 1: Classify objects with depth validation
        detections = self.classifier.classify_with_depth_validation(rgb_frame, depth_map)
        
        # Step 2: Track objects across frames
        track_ids = self.tracker.update(detections, self.frame_count)
        tracked_objects = self.tracker.get_tracked_objects()
        
        # Step 3: Prioritize based on context
        prioritized = self.prioritizer.prioritize_obstacles(tracked_objects, rgb_frame.shape[1])
        
        # Step 4: Generate audio cues
        audio_cues = self.audio_mapper.generate_audio_cues(prioritized)
        
        # Step 5: Check for critical alerts
        critical_alert = self._get_critical_alert(prioritized)
        
        return {
            'raw_detections': detections,
            'tracked_objects': tracked_objects,
            'prioritized_obstacles': prioritized,
            'audio_cues': audio_cues,
            'critical_alert': critical_alert,
            'visualization_frame': self.classifier.visualize_detections(rgb_frame, detections)
        }
    
    def _get_critical_alert(self, prioritized_obstacles):
        """Check for critical situations needing immediate attention"""
        for obstacle in prioritized_obstacles[:3]:  # Top 3 most critical
            if (obstacle.get('is_urgent', False) and 
                obstacle['zone'] == 'center' and 
                obstacle['depth'] < 1.5):
                return f"CRITICAL: {obstacle['class']} at {obstacle['depth']:.1f}m - {obstacle['suggested_action']}"
        return None

# Example usage
def main():
    # Initialize pipeline with your trained YOLO model
    pipeline = ObjectClassificationPipeline('runs/detect/yolov8n_coco_exp_01/weights/best.pt')
    
    # Example: Process webcam with simulated depth
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, rgb_frame = cap.read()
        if not ret:
            break
        
        # Simulate depth map (replace with actual Depth Anything V2 output)
        depth_map = np.random.rand(rgb_frame.shape[0], rgb_frame.shape[1]) * 5.0
        
        # Process frame
        results = pipeline.process_frame(rgb_frame, depth_map)
        
        # Display results
        cv2.imshow('Object Classification', results['visualization_frame'])
        
        # Print updates every 30 frames
        if pipeline.frame_count % 30 == 0:
            pipeline.prioritizer.print_prioritized_obstacles(results['prioritized_obstacles'])
            pipeline.audio_mapper.print_audio_cues(results['audio_cues'])
            
            if results['critical_alert']:
                print(f"\n🚨 {results['critical_alert']}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()