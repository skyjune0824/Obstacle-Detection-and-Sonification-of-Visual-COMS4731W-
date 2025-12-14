import numpy as np

class ContextAwarePrioritizer:
    def __init__(self):
        self.zone_weights = {'left': 1.0, 'center': 1.5, 'right': 1.0}
    
    def prioritize_obstacles(self, detections, frame_width):
        """Prioritize obstacles based on multiple factors"""
        prioritized = []
        
        for det in detections:
            zone = self._get_spatial_zone(det['bbox'], frame_width)
            threat_score = self._calculate_threat_score(det, zone)
            suggested_action = self._get_suggested_action(det, zone)
            
            prioritized.append({
                **det,
                'zone': zone,
                'threat_score': threat_score,
                'suggested_action': suggested_action
            })
        
        # Sort by threat score (descending)
        return sorted(prioritized, key=lambda x: x['threat_score'], reverse=True)
    
    def _calculate_threat_score(self, detection, zone):
        """Calculate comprehensive threat score"""
        # Base priority from class
        class_score = detection['priority']
        
        # Distance factor (closer = higher threat)
        distance_score = 0
        if detection['depth'] is not None:
            distance_score = 10.0 / (detection['depth'] + 0.1)
        
        # Zone importance
        zone_score = self.zone_weights[zone]
        
        # Confidence factor
        confidence_score = detection['confidence'] * 5
        
        # Stability bonus for tracked objects
        stability_bonus = 3 if detection.get('is_stable', False) else 0
        
        # Urgency bonus
        urgency_bonus = 20 if detection.get('is_urgent', False) else 0
        
        return (class_score + distance_score + zone_score + 
                confidence_score + stability_bonus + urgency_bonus)
    
    def _get_spatial_zone(self, bbox, frame_width):
        """Determine which spatial zone the object is in"""
        x_center = (bbox[0] + bbox[2]) // 2
        left_bound = frame_width // 3
        right_bound = 2 * frame_width // 3
        
        if x_center < left_bound:
            return 'left'
        elif x_center < right_bound:
            return 'center'
        else:
            return 'right'
    
    def _get_suggested_action(self, detection, zone):
        """Get suggested navigation action based on object context"""
        if not detection.get('is_urgent', False):
            return "continue"
        
        if zone == 'center':
            if detection['class'] in ['person', 'bicycle']:
                return "slow_down"
            else:
                return "avoid"
        elif zone == 'left':
            return "veer_right"
        elif zone == 'right':
            return "veer_left"
        
        return "caution"
    
    def print_prioritized_obstacles(self, prioritized_obstacles, top_k=3):
        """Print top obstacles for debugging"""
        print("\n" + "="*50)
        print("PRIORITIZED OBSTACLES")
        print("="*50)
        
        for i, obstacle in enumerate(prioritized_obstacles[:top_k]):
            urgency = "⚡ URGENT" if obstacle.get('is_urgent', False) else "⚠️ WARNING"
            print(f"{i+1}. {obstacle['class'].upper()} ({obstacle['navigation_class']})")
            print(f"   Distance: {obstacle['depth']:.1f}m | Zone: {obstacle['zone']} | {urgency}")
            print(f"   Action: {obstacle['suggested_action']} | Score: {obstacle['threat_score']:.1f}")
            print()