import cv2
import numpy as np
from collections import deque

class SegmentationModule:
    """
    Obstacle Segmentation and Mapping Module for Visual Navigation
    Processes depth maps to detect obstacles, create occupancy grids, and identify moving objects
    """
    
    def __init__(self, depth_threshold=2.0, grid_resolution=0.1, grid_size=(100, 100)):
        """
        Initialize segmentation module
        
        Args:
            depth_threshold: Maximum distance (meters) to consider as obstacle
            grid_resolution: Size of each grid cell in meters (e.g., 0.1 = 10cm)
            grid_size: Tuple (rows, cols) for occupancy grid dimensions
        """
        self.depth_threshold = depth_threshold
        self.grid_resolution = grid_resolution
        self.grid_size = grid_size
        
        # Initialize occupancy grid (0 = free, 255 = occupied)
        self.occupancy_grid = np.zeros(grid_size, dtype=np.uint8)
        
        # Store previous depth map for motion detection
        self.prev_depth = None
        self.prev_gray = None
        
        # Optical flow parameters
        self.flow_history = deque(maxlen=5)
        
    def process_depth_map(self, depth_map):
        """
        Main processing pipeline for depth map
        
        Args:
            depth_map: Input depth map (H x W), values in meters
            
        Returns:
            dict containing:
                - zones: obstacle info for left/center/right zones
                - occupancy_grid: 2D grid representation
                - moving_obstacles: list of moving obstacle locations
        """
        # Step 1: Threshold depth map to identify obstacles
        obstacle_mask = self.threshold_depth_map(depth_map)
        
        # Step 2: Divide into spatial zones
        zones = self.divide_into_zones(depth_map, obstacle_mask)
        
        # Step 3: Create occupancy grid
        self.occupancy_grid = self.create_occupancy_grid(depth_map, obstacle_mask)
        
        # Step 4: Detect moving obstacles
        moving_obstacles = self.detect_moving_obstacles(depth_map)
        
        return {
            'obstacle_mask': obstacle_mask,
            'zones': zones,
            'occupancy_grid': self.occupancy_grid,
            'moving_obstacles': moving_obstacles
        }
    
    def threshold_depth_map(self, depth_map):
        """
        Apply thresholding to separate foreground obstacles from background
        
        Args:
            depth_map: Input depth map (H x W) in meters
            
        Returns:
            Binary mask where 255 = obstacle, 0 = free space
        """
        # Normalize depth map to 0-255 for processing
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Create binary mask: obstacles are closer than threshold
        obstacle_mask = (depth_map > 0) & (depth_map < self.depth_threshold)
        obstacle_mask = obstacle_mask.astype(np.uint8) * 255
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        
        return obstacle_mask
    
    def divide_into_zones(self, depth_map, obstacle_mask):
        """
        Divide depth map into left, center, right zones and analyze each
        
        Args:
            depth_map: Input depth map (H x W)
            obstacle_mask: Binary obstacle mask
            
        Returns:
            Dictionary with zone information
        """
        height, width = depth_map.shape
        zone_width = width // 3
        
        zones = {}
        zone_names = ['left', 'center', 'right']
        
        for i, zone_name in enumerate(zone_names):
            # Extract zone region
            start_col = i * zone_width
            end_col = (i + 1) * zone_width if i < 2 else width
            
            zone_depth = depth_map[:, start_col:end_col]
            zone_mask = obstacle_mask[:, start_col:end_col]
            
            # Calculate zone statistics
            obstacle_pixels = np.sum(zone_mask > 0)
            total_pixels = zone_mask.size
            obstacle_density = obstacle_pixels / total_pixels
            
            # Find closest obstacle in zone
            valid_depths = zone_depth[(zone_mask > 0) & (zone_depth > 0)]
            min_distance = np.min(valid_depths) if len(valid_depths) > 0 else float('inf')
            
            zones[zone_name] = {
                'obstacle_density': obstacle_density,
                'min_distance': min_distance,
                'is_clear': obstacle_density < 0.1,  # Less than 10% occupied
                'num_obstacles': obstacle_pixels
            }
        
        return zones
    
    def create_occupancy_grid(self, depth_map, obstacle_mask):
        """
        Create 2D bird's-eye view occupancy grid from depth map
        
        Args:
            depth_map: Input depth map (H x W) in meters
            obstacle_mask: Binary obstacle mask
            
        Returns:
            2D occupancy grid (grid_size), values 0-255
        """
        height, width = depth_map.shape
        grid = np.zeros(self.grid_size, dtype=np.uint8)
        
        # Iterate through depth map pixels
        for y in range(height):
            for x in range(width):
                if obstacle_mask[y, x] > 0 and depth_map[y, x] > 0:
                    depth = depth_map[y, x]
                    
                    # Project to world coordinates (simplified projection)
                    # Assume camera is at origin, looking forward
                    world_x = (x - width / 2) * depth / width * 2  # Lateral position
                    world_z = depth  # Forward distance
                    
                    # Convert to grid coordinates
                    grid_x = int((world_x / self.grid_resolution) + self.grid_size[1] / 2)
                    grid_z = int(world_z / self.grid_resolution)
                    
                    # Mark grid cell as occupied if within bounds
                    if 0 <= grid_x < self.grid_size[1] and 0 <= grid_z < self.grid_size[0]:
                        grid[grid_z, grid_x] = 255
        
        return grid
    
    def detect_moving_obstacles(self, depth_map):
        """
        Detect moving obstacles using temporal depth difference and optical flow
        
        Args:
            depth_map: Current depth map (H x W)
            
        Returns:
            List of moving obstacle locations (x, y, motion_magnitude)
        """
        moving_obstacles = []
        
        if self.prev_depth is None:
            self.prev_depth = depth_map.copy()
            return moving_obstacles
        
        # Method 1: Temporal depth difference
        depth_diff = cv2.absdiff(depth_map, self.prev_depth)
        motion_mask = (depth_diff > 0.1).astype(np.uint8) * 255  # 10cm threshold
        
        # Method 2: Optical flow on depth map
        # Convert depth maps to uint8 for optical flow
        prev_depth_norm = cv2.normalize(self.prev_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        curr_depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_depth_norm, 
            curr_depth_norm,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Compute flow magnitude
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Threshold to find significant motion (moving obstacles)
        motion_threshold = 2.0  # pixels
        significant_motion = magnitude > motion_threshold
        
        # Combine with depth difference for robust detection
        combined_motion = (significant_motion.astype(np.uint8) * motion_mask) > 0
        
        # Find contours of moving regions
        contours, _ = cv2.findContours(combined_motion.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract moving obstacle locations
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Minimum area threshold
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    avg_magnitude = np.mean(magnitude[combined_motion])
                    
                    moving_obstacles.append({
                        'x': cx,
                        'y': cy,
                        'motion_magnitude': avg_magnitude,
                        'depth': depth_map[cy, cx] if depth_map[cy, cx] > 0 else None
                    })
        
        # Update previous depth map
        self.prev_depth = depth_map.copy()
        
        return moving_obstacles
    
    def visualize_results(self, depth_map, results, rgb_frame=None):
        """
        Visualize segmentation results for debugging
        
        Args:
            depth_map: Original depth map
            results: Output from process_depth_map()
            rgb_frame: Optional RGB frame to overlay
            
        Returns:
            Visualization image
        """
        # Create visualization canvas
        if rgb_frame is not None:
            vis = rgb_frame.copy()
        else:
            depth_colored = cv2.applyColorMap(
                cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            vis = depth_colored
        
        height, width = vis.shape[:2]
        zone_width = width // 3
        
        # Draw zone boundaries
        cv2.line(vis, (zone_width, 0), (zone_width, height), (0, 255, 0), 2)
        cv2.line(vis, (2 * zone_width, 0), (2 * zone_width, height), (0, 255, 0), 2)
        
        # Label zones with statistics
        zones = results['zones']
        zone_names = ['LEFT', 'CENTER', 'RIGHT']
        for i, zone_name in enumerate(['left', 'center', 'right']):
            zone_info = zones[zone_name]
            x_pos = int((i + 0.5) * zone_width)
            
            status = "CLEAR" if zone_info['is_clear'] else "BLOCKED"
            color = (0, 255, 0) if zone_info['is_clear'] else (0, 0, 255)
            
            text = f"{zone_names[i]}: {status}"
            cv2.putText(vis, text, (x_pos - 50, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            dist_text = f"{zone_info['min_distance']:.2f}m"
            cv2.putText(vis, dist_text, (x_pos - 40, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw moving obstacles
        for obs in results['moving_obstacles']:
            cv2.circle(vis, (obs['x'], obs['y']), 10, (255, 0, 255), -1)
            cv2.putText(vis, "MOVING", (obs['x'] + 15, obs['y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Create occupancy grid visualization
        grid_vis = cv2.resize(results['occupancy_grid'], (200, 200), 
                             interpolation=cv2.INTER_NEAREST)
        grid_vis = cv2.applyColorMap(grid_vis, cv2.COLORMAP_BONE)
        
        return vis, grid_vis


def main():
    """
    Example usage with video file or webcam
    """
    # Initialize segmentation module
    segmenter = SegmentationModule(
        depth_threshold=2.0,  # 2 meters
        grid_resolution=0.1,   # 10cm cells
        grid_size=(100, 100)   # 10m x 10m grid
    )
    
    # Example: Process depth maps
    # Replace this with actual depth map source
    cap = cv2.VideoCapture(0)  # Or video file path
    
    while True:
        ret, rgb_frame = cap.read()
        if not ret:
            break
        
        # TODO: Get depth map
        # depth_map = depth_estimation_model.predict(rgb_frame)
        # For demo purposes, create dummy depth map
        depth_map = np.random.rand(*rgb_frame.shape[:2]) * 3.0  # Simulated depth
        
        # Process depth map
        results = segmenter.process_depth_map(depth_map)
        
        # Visualize results
        vis, grid_vis = segmenter.visualize_results(depth_map, results, rgb_frame)
        
        # Display
        cv2.imshow('Segmentation Results', vis)
        cv2.imshow('Occupancy Grid', grid_vis)
        
        # Print zone information
        print("\n=== Zone Analysis ===")
        for zone_name, zone_info in results['zones'].items():
            print(f"{zone_name.upper()}: "
                  f"Clear={zone_info['is_clear']}, "
                  f"Min Dist={zone_info['min_distance']:.2f}m, "
                  f"Density={zone_info['obstacle_density']:.2%}")
        
        if len(results['moving_obstacles']) > 0:
            print(f"\nDetected {len(results['moving_obstacles'])} moving obstacles!")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
