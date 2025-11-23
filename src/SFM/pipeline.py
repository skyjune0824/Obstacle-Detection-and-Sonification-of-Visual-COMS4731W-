# Debug Setting
from config import DEBUG

# Video Processing
import cv2
from PIL import Image

# Modules
from src.AudioSynthesis.synthesize import SpatialAudioFeedback

class SFM_Pipeline:
    """ Structure From Motion Pipeline

    Takes video, sampling key frames, to perform sparse structure from motion for
    estimating object distance for sonification.
    """

    def __init__(self, rate):
        self.synth = SpatialAudioFeedback()
        self.sample_rate = rate

    def pipeline(self, source):
        """
        The following function marks the pipeline for creating a minimal point cloud
        from multiple images in motion, performing segmentation, and synthesizing spatial audio.

        Source: Path to video we desire to perform our pipeline on.
        Debug: Toggles Verbose

        """

        print(f"Performing SFM on video located at: {source}")

        # Open Video Path
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"Error: Could not reach video signal {source}")
        
        # Establish Frame Count
        frame_cnt = 0

        # Core Pipeline Loop
        while True:
            # Read Frame While Possible
            ret, frame_one = cap.read()
            ret, frame_two = cap.read()
            if not ret:
                break

            # Sample and Process Frame in multiples of "Sample Rate". 
            if frame_cnt % self.sample_rate == 0:
                self.process(frame_one, frame_two)

            # Increment Frame Count
            frame_cnt += 1
            
        # Release Video Source
        cap.release()

        log("Complete.")

        # return self.audio_trace
    
    def process(self, frame_one, frame_two):
        """ Process
        Denotes one singular pass in the pipeline.
        1. Captures Depth Map
        2. Segments Depth Map and creates BEV.
        3. Synthesizes into Spatial Audio.
        
        """

        # Preprocess
        # Extract ORB Features
        orb = cv2.ORB_create(nfeatures=5000)

        # Get Key Points Between Images
        keypoint_one, descriptors_one = orb.detectAndCompute(frame_one, None)
        keypoint_two, descriptors_two = orb.detectAndCompute(frame_two, None)

        # Match Descriptors Between Images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_one, descriptors_two)

        # Sort Matches
        matches = sorted(matches, key=lambda x: x.distance)

        if DEBUG:
            match_img = cv2.drawMatches(
                frame_one, keypoint_one,
                frame_two, keypoint_two,
                matches[:50],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imshow("Feature Matches", match_img)
            cv2.waitKey(0)


    # def track_features()


def log(msg):
    if DEBUG:
        print(msg)