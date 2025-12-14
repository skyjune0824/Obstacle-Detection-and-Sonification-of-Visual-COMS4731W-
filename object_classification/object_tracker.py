import numpy as np
from collections import defaultdict, deque

class ObjectTracker:
    def __init__(self, max_age=10, max_distance=50):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.max_distance = max_distance
        
    def update(self, detections, frame_count):
        """Update tracks with new detections"""
        current_ids = []
        
        for det in detections:
            centroid = self._get_centroid(det['bbox'])
            track_id = self._find_matching_track(centroid, det['class'])
            
            if track_id is not None:
                # Update existing track
                self.tracks[track_id]['centroid'] = centroid
                self.tracks[track_id]['bbox'] = det['bbox']
                self.tracks[track_id]['depth'] = det['depth']
                self.tracks[track_id]['confidence'] = det['confidence']
                self.tracks[track_id]['last_seen'] = frame_count
                self.tracks[track_id]['confidence_history'].append(det['confidence'])
                self.tracks[track_id]['depth_history'].append(det['depth'])
                
                # Keep only recent history
                if len(self.tracks[track_id]['confidence_history']) > 10:
                    self.tracks[track_id]['confidence_history'].pop(0)
                    self.tracks[track_id]['depth_history'].pop(0)
                    
            else:
                # Create new track
                self.tracks[self.next_id] = {
                    'centroid': centroid,
                    'bbox': det['bbox'],
                    'depth': det['depth'],
                    'class': det['class'],
                    'navigation_class': det['navigation_class'],
                    'first_seen': frame_count,
                    'last_seen': frame_count,
                    'confidence_history': deque([det['confidence']], maxlen=10),
                    'depth_history': deque([det['depth']], maxlen=10),
                    'priority': det['priority'],
                    'safe_distance': det['safe_distance']
                }
                track_id = self.next_id
                self.next_id += 1
            
            current_ids.append(track_id)
        
        # Remove stale tracks
        self._remove_stale_tracks(frame_count)
        
        return current_ids
    
    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _find_matching_track(self, centroid, class_name):
        best_track_id = None
        best_distance = float('inf')
        
        for track_id, track in self.tracks.items():
            if track['class'] != class_name:
                continue
            
            prev_centroid = track['centroid']
            distance = np.sqrt((centroid[0] - prev_centroid[0])**2 + 
                             (centroid[1] - prev_centroid[1])**2)
            
            if distance < self.max_distance and distance < best_distance:
                best_distance = distance
                best_track_id = track_id
        
        return best_track_id
    
    def _remove_stale_tracks(self, frame_count):
        """Remove tracks that haven't been seen recently"""
        stale_tracks = []
        for track_id, track in self.tracks.items():
            if frame_count - track['last_seen'] > self.max_age:
                stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            del self.tracks[track_id]
    
    def get_tracked_objects(self):
        """Get objects with tracking information"""
        tracked_objects = []
        for track_id, track in self.tracks.items():
            avg_confidence = np.mean(list(track['confidence_history']))
            avg_depth = np.mean(list(track['depth_history']))
            
            tracked_objects.append({
                'track_id': track_id,
                'class': track['class'],
                'navigation_class': track['navigation_class'],
                'bbox': track['bbox'],
                'depth': avg_depth,
                'confidence': avg_confidence,
                'age': track['last_seen'] - track['first_seen'],
                'is_stable': len(track['confidence_history']) > 3,
                'priority': track['priority'],
                'safe_distance': track['safe_distance'],
                'is_urgent': avg_depth < track['safe_distance']
            })
        
        return tracked_objects