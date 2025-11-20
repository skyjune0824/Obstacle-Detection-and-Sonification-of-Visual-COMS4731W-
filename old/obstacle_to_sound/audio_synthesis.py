import pyaudio
import numpy as np
import threading

class SpatialAudioFeedback:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=2,  # Stereo output
            rate=sample_rate,
            output=True,
            frames_per_buffer=1024
        )
        self.is_playing = False
        
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
        stereo[:, 0] = tone * left_gain   # Left channel
        stereo[:, 1] = tone * right_gain  # Right channel
        
        return stereo
    
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
            if params['frequency'] > 0:  # Skip zones without obstacles
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
        
        # Play audio
        self.stream.write(mixed_audio.tobytes())
    
    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
