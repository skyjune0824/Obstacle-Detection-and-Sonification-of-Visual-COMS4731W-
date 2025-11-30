import pyaudio
import numpy as np
import threading

class SpatialAudioFeedback:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.frames_per_buffer = 1024
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=2, 
            rate=sample_rate,
            output=True,
            frames_per_buffer=self.frames_per_buffer
        )

        # Shared Buffer that holds our current playing tone
        self.current_tone = np.zeros((self.frames_per_buffer, 2), dtype=np.float32)

        # Mutex Lock to Ensure Only one player has access to our shared tone buffer
        self.lock = threading.Lock()
        self.running = True

        # Initiates Looping Audio Thread
        self.thread = threading.Thread(target=self.audio_loop)
        self.thread.start()

    
    def audio_loop(self):
        """ Audio Loop
        
        While the stream is active, if we have a lock free, we can write an audio segment to the stream until
        a new tone arrives.
        """

        while self.running:
            with self.lock:
                chunk = self.current_tone.copy()
            
            self.stream.write(chunk.tobytes(), exception_on_underflow=False)

        
    def generate_tone(self, frequency, duration, volume, pan):
        """
        Generate stereo audio tone with spatial panning
        
        Args:
            frequency: Tone frequency in Hz
            duration: Duration in seconds
            volume: Amplitude (0-1)
            pan: Stereo position (-1 left, 0 center, +1 right)
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Generate sine wave
        tone = volume * np.sin(2 * np.pi * frequency * t)
        
        # Apply stereo panning using constant-power panning
        # https://www.cs.cmu.edu/~music/icm-online/readings/panlaws/
        pan_rad = (pan + 1) * np.pi / 4  # Convert [-1, 1] to [0, pi/2]
        left_gain = np.cos(pan_rad)
        right_gain = np.sin(pan_rad)
        
        # Create stereo signal
        stereo = np.zeros((len(tone), 2), dtype=np.float32)
        stereo[:, 0] = tone * left_gain  
        stereo[:, 1] = tone * right_gain 
        
        return stereo
    
    def obstacle_to_audio_params(self, zones):
        """
        Convert obstacle zones to audio parameters
        
        Args:
            zones: Spatial zone information our segmentation output.
        
        Returns:
            audio_params: Dict with frequency, volume, pan per zone
        """
        audio_params = {}
        
        # Base frequency and volume ranges
        BASE_FREQ = 200  
        MAX_FREQ = 800 
        BASE_VOLUME = 0.2
        MAX_VOLUME = 0.8
        
        for zone_name, zone_data in zones.items():
            if 'min_distance' in zone_data:
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
            elif 'obstacle_density' in zone_data:
                density = zone_data['obstacle_density']

                if density == 0:
                    # No obstacles in this zone
                    audio_params[zone_name] = {
                        'frequency': 0,
                        'volume': 0,
                        'pan': 0
                    }
                    continue

                # Map density to frequency (more = higher pitch)
                # Log Scale. 
                freq = BASE_FREQ + (MAX_FREQ - BASE_FREQ) * (np.log1p(density) / np.log1p(density + 1))
                volume = BASE_VOLUME + (MAX_VOLUME - BASE_VOLUME) * (np.log1p(density) / np.log1p(density + 1))

                # Stereo panning: -1 (left) to +1 (right)
                pan_map = {'left': -0.8, 'center': 0.0, 'right': 0.8}
                pan = pan_map[zone_name]
                
                audio_params[zone_name] = {
                    'frequency': freq,
                    'volume': volume,
                    'pan': pan
                }
        
        return audio_params

    
    def play_audio_feedback(self, audio_params, duration=0.2):
        """
        Play real-time spatial audio feedback for all zones
        
        Args:
            audio_params: Dict from obstacle_to_audio_params()
            duration: Tone duration in seconds
        """
        # Mix audio from all three zones
        mixed_audio = np.zeros((int(self.sample_rate * duration), 2), dtype=np.float32)
        
        for zone_name, params in audio_params.items():
            # Skip zones without obstacles
            if params['frequency'] > 0: 
                tone = self.generate_tone(
                    params['frequency'],
                    duration,
                    params['volume'],
                    params['pan']
                )
                mixed_audio += tone
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 1.0:
            mixed_audio /= max_val
        
        # Prepare one buffer-sized chunk
        chunk_len = self.frames_per_buffer
        if mixed_audio.shape[0] >= chunk_len:
            chunk = mixed_audio[:chunk_len]
        else:
            # Loop Current Audio to fill shared buffer.
            reps = (chunk_len // mixed_audio.shape[0]) + 1
            chunk[:mixed_audio.shape[0]] = mixed_audio

        # Update the live tone chunk if we have a lock free
        with self.lock:
            self.current_tone = chunk.astype(np.float32)
    
    def close(self):
        # Send silence into buffer
        self.stream.stop_stream()
        self.stream.close()
        self.running = False
        self.thread.join()
        self.p.terminate()
