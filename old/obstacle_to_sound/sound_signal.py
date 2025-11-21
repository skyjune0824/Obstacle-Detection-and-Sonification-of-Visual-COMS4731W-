import numpy as np

def process_obstacles(detection_output, depth_map, frame_width):
    """
    Extract obstacle information for audio mapping
    
    Args:
        detection_output: From FusionVision or RFNet
        depth_map: Depth values from your MDE model
        frame_width: Width of the video frame
    
    Returns:
        obstacle_zones: Dict with spatial zone information
    """
    # Define spatial zones (left, center, right)
    left_bound = frame_width // 3
    right_bound = 2 * frame_width // 3
    
    # Distance thresholds from your proposal
    URGENT_ZONE = 1.0   # meters
    WARNING_ZONE = 2.0  # meters
    SAFE_ZONE = 5.0     # meters
    
    zones = {
        'left': {'min_distance': float('inf'), 'objects': []},
        'center': {'min_distance': float('inf'), 'objects': []},
        'right': {'min_distance': float('inf'), 'objects': []}
    }
    
    # Process each detected object (for FusionVision output)
    for obj in detection_output:
        bbox_center_x = (obj['bbox_2d'][0] + obj['bbox_2d'][2]) / 2
        distance = obj['distance']
        
        # Determine spatial zone
        if bbox_center_x < left_bound:
            zone = 'left'
        elif bbox_center_x < right_bound:
            zone = 'center'
        else:
            zone = 'right'
        
        # Update minimum distance for zone
        zones[zone]['min_distance'] = min(zones[zone]['min_distance'], distance)
        zones[zone]['objects'].append({
            'class': obj['class'],
            'distance': distance,
            'urgency': 'urgent' if distance < URGENT_ZONE else 
                      'warning' if distance < WARNING_ZONE else 'safe'
        })
    
    return zones
def obstacle_to_audio_params(zones):
    """
    Convert obstacle zones to audio parameters
    
    Args:
        zones: Spatial zone information from process_obstacles()
    
    Returns:
        audio_params: Dict with frequency, volume, pan per zone
    """
    audio_params = {}
    
    # Base frequency and volume ranges
    BASE_FREQ = 200  # Hz
    MAX_FREQ = 2000  # Hz
    BASE_VOLUME = 0.2
    MAX_VOLUME = 1.0
    
    for zone_name, zone_data in zones.items():
        distance = zone_data['min_distance']
        
        if distance == float('inf'):
            # No obstacles in this zone
            audio_params[zone_name] = {
                'frequency': 0,
                'volume': 0,
                'pan': 0
            }
            continue
        
        # Map distance to frequency (closer = higher pitch)
        # Inverse exponential mapping for more better feedback
        freq = BASE_FREQ + (MAX_FREQ - BASE_FREQ) * np.exp(-distance / 2.0)
        
        # Map distance to volume (closer = louder)
        volume = BASE_VOLUME + (MAX_VOLUME - BASE_VOLUME) * (1 - distance / 5.0)
        volume = np.clip(volume, BASE_VOLUME, MAX_VOLUME)
        
        # Stereo panning: -1 (left) to +1 (right)
        pan_map = {'left': -0.8, 'center': 0.0, 'right': 0.8}
        pan = pan_map[zone_name]
        
        audio_params[zone_name] = {
            'frequency': freq,
            'volume': volume,
            'pan': pan
        }
    
    return audio_params
