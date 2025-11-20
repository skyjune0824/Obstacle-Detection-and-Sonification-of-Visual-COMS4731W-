import numpy as np
import sounddevice as sd

class AudioSynthesis:
    """ AudioSynthesis

    Converts occupancy grid and zones into spatial panning audio,
    distinguishing obstacle distance by pitch.
    """

    def __init__(self, sample_rate=44100, buffer_size=1024):
        self.fs = sample_rate
        self.buffer_size = buffer_size
        self.frequency = 440
        self.volume = 0.5
        self.pan = 0.0
        self.phase = 0

        # Audio Stream
        self.stream = sd.OutputStream(
            channels=2,
            samplerate=self.fs,
            callback=self.output_audio,
            blocksize=self.buffer_size
        )

        self.stream.start()


    def zones_to_audio(self, zones):
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

        for zone, data in zones.items():
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
            
            # DEBUG
            print(f"FREQ: {freq}, VOL: {vol}, PAN: {pan_map[zone]}")

            params[zone] = {
                'frequency': freq,
                'volume': vol,
                'pan': pan_map[zone]
            }
        
        return params

    def output_audio(self, outdata, frames, time, status):
        """ Output Audio

        Play short stereo audio with panning. 
        """

        # Time Array
        t = (np.arange(frames) + self.phase) / self.fs
        self.phase += frames

        # Sine Wave
        tone = np.sin(2 * np.pi * self.frequency * t) * self.volume

        # Stereo Panning
        # Normalization
        normalized_pan = (self.pan + 1)/2
        left = tone * np.sqrt(1 - normalized_pan)
        right = tone * np.sqrt(normalized_pan)

        # Channel Combination
        outdata[:] = np.column_stack([left, right])


    def continuous_audio(self, params):
        """ Continuous Audio
        
        Plays a continuous stream of audio to match obstacle information.
        """

        for zone, data in params.items():
            # Update Audio Params for Continuous Stream
            if data['frequency'] is not None:
                self.frequency = data['frequency']
            if data['volume'] is not None:
                self.volume = data['volume']
            if data['pan'] is not None:
                self.pan = data['pan']

    def stop(self):
        self.stream.stop()
        self.stream.close()





        