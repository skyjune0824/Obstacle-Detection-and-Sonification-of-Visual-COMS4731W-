
import torch
from segmentation_module import SegmentationModule

def integrate_with_depth_model():
    """
    Example showing how to connect with depth estimation output
    """
    # Initialize your teammate's depth model
    # Example with Depth Anything V2
    # depth_model = torch.hub.load('depth-anything/Depth-Anything-V2', 'vitl')
    
    # Initialize segmentation module
    segmenter = SegmentationModule(
        depth_threshold=2.5,
        grid_resolution=0.1,
        grid_size=(100, 100)
    )
    
    # Video processing pipeline
    cap = cv2.VideoCapture('input_video.mp4')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Step 1: Get depth map from teammate's model
        # depth_map = depth_model.infer_image(frame)  # Returns depth in meters
        
        # For demonstration (replace with actual model output)
        depth_map = np.random.rand(*frame.shape[:2]) * 5.0
        
        # Step 2: Process with segmentation module
        seg_results = segmenter.process_depth_map(depth_map)
        
        # Step 3: Use results for navigation decisions
        zones = seg_results['zones']
        
        # Navigation logic example
        if not zones['center']['is_clear']:
            if zones['left']['is_clear']:
                print("ALERT: Move LEFT")
            elif zones['right']['is_clear']:
                print("ALERT: Move RIGHT")
            else:
                print("ALERT: STOP - All zones blocked")
        
        # Check for moving obstacles
        if len(seg_results['moving_obstacles']) > 0:
            for obs in seg_results['moving_obstacles']:
                if obs['depth'] and obs['depth'] < 1.5:
                    print(f"WARNING: Moving obstacle at {obs['depth']:.1f}m!")
        
        # Visualize
        vis, grid_vis = segmenter.visualize_results(depth_map, seg_results, frame)
        cv2.imshow('Navigation View', vis)
        cv2.imshow('Occupancy Grid', grid_vis)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    integrate_with_depth_model()
