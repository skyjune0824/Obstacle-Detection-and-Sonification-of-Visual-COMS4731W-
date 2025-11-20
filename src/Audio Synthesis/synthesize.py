import numpy as np
import sounddevice as sd

class AudioSynthesis:
    """ AudioSynthesis

    Converts occupancy grid and zones into spatial panning audio,
    distinguishing obstacle distance by pitch.
    """
    def __init__(self):
        return NotImplemented
    
    def zones_to_audio(self, zone):
        """ Zones to Audio

        Takes divided zones and creates parametrizes audio signal. 
        """

        # Store Audio parameters for returning
        params = {}
 
        # Furthest Object Freq
        base_freq = 200
        # Closest Object Freq
        max_freq = 1000
        
        # Common Freq
        freq = 0
        # Common Volume
        vol = 0

        # Min Distance (In meters)
        min_dist = 0.2 
        max_dist = 5.0 

        # Panning Map, Looks like avoiding "1" avoids the sounds from being directly at the edge of each speaker.
        pan_map = {'left': -0.8, 'center': 0.0, 'right': 0.8}

        for zone, data in zone.items():
            dist = data["min_distance"]
            density = data["obstacle_density"]

            # Zero Obstacles = No Audible Frequency
            if min_dist == float('inf'):
                freq = 0
                vol = 0
            else:
                # Map obstacle distance to frequency 
                normalized_dist = ((dist - min_dist) / max_dist - min_dist)
                freq = max_freq - (normalized_dist * (max_freq - base_freq))

                # Clip Freq in Case
                freq = np.clip(freq, base_freq, max_freq)

                # Volume Mapping from 0 - 1 where density already is a fraction.
                vol = np.clip(density, 0.1, 1.0)

            params[zone] = {
                'frequency': freq,
                'volume': vol,
                'pan': pan_map[zone]
            }
        
        return params

    def


        