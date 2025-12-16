import numpy as np

class ObjectAwareAudioMapper:
    def __init__(self):
        self.object_sound_profiles = {
            'pedestrian': {
                'base_freq': 400,
                'waveform': 'sine',
                'pulse_rate': 2.0
            },
            'vehicle': {
                'base_freq': 300, 
                'waveform': 'square',
                'pulse_rate': 1.5
            },
            'bicycle': {
                'base_freq': 350,
                'waveform': 'sawtooth', 
                'pulse_rate': 2.5
            },
            'furniture': {
                'base_freq': 200,
                'waveform': 'sine',
                'pulse_rate': 1.0
            }
        }
        
        self.zone_panning = {'left': -0.8, 'center': 0.0, 'right': 0.8}
    
    def generate_audio_cues(self, prioritized_obstacles):
        """Generate audio parameters based on object classification"""
        audio_cues = {}
        
        for zone in ['left', 'center', 'right']:
            zone_obstacles = [obs for obs in prioritized_obstacles if obs['zone'] == zone]
            
            if not zone_obstacles:
                audio_cues[zone] = self._get_silent_cue(zone)
            else:
                critical_obstacle = zone_obstacles[0]
                audio_cues[zone] = self._map_obstacle_to_audio(critical_obstacle, zone)
        
        return audio_cues
    
    def _map_obstacle_to_audio(self, obstacle, zone):
        """Map obstacle properties to audio parameters"""
        sound_profile = self.object_sound_profiles.get(
            obstacle['navigation_class'], 
            self.object_sound_profiles['furniture']
        )
        
        distance = obstacle['depth']
        
        # Frequency modulation based on distance and object type
        base_freq = sound_profile['base_freq']
        freq_modulation = 800 / (distance + 0.1)  # Higher for closer objects
        frequency = min(base_freq + freq_modulation, 2000)
        
        # Volume modulation
        base_volume = 0.3
        volume_modulation = 0.7 / (distance + 0.1)
        volume = min(base_volume + volume_modulation, 1.0)
        
        # Apply urgency boost
        if obstacle.get('is_urgent', False):
            volume = min(volume * 1.3, 1.0)
            frequency = min(frequency * 1.2, 2000)
        
        # Pulse rate modulation
        pulse_rate = sound_profile['pulse_rate'] * (2.0 / (distance + 0.1))
        
        return {
            'frequency': frequency,
            'volume': volume,
            'pan': self.zone_panning[zone],
            'waveform': sound_profile['waveform'],
            'pulse_rate': min(pulse_rate, 5.0),  # Max 5 Hz
            'object_type': obstacle['navigation_class'],
            'urgency': 'high' if obstacle.get('is_urgent', False) else 'medium',
            'active': True
        }
    
    def _get_silent_cue(self, zone):
        """Return silent audio cue for empty zones"""
        return {
            'frequency': 0,
            'volume': 0,
            'pan': self.zone_panning[zone],
            'waveform': 'sine',
            'pulse_rate': 0,
            'object_type': 'none',
            'urgency': 'none',
            'active': False
        }
    
    def print_audio_cues(self, audio_cues):
        """Print audio cues for debugging"""
        print("\n" + "="*40)
        print("AUDIO CUES")
        print("="*40)
        
        for zone, cue in audio_cues.items():
            if cue['active']:
                status = f"🎵 {cue['object_type'].upper()} - {cue['frequency']:.0f}Hz"
                if cue['urgency'] == 'high':
                    status += " ⚡"
                print(f"{zone.upper():6}: {status}")
            else:
                print(f"{zone.upper():6}: 🔇 Clear")