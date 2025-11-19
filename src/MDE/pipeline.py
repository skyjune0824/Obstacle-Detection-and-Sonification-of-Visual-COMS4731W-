# Debug Setting
from config import DEBUG

# Video Processing
import cv2

# Image Processing
from PIL import Image

# Modules
from src.MDE.estimator import MDE
from src.Segmentation.segmentation import SegmentationModule

class MDE_Pipeline:
    """
    Monocular Depth Estimation Pipeline

    Processes a video into spatial audio for obstacle detection via monocular depth estimation,
    segmentation, and spatial audio synthesis.
    """

    def __init__(self, rate):
        self.model = MDE()
        self.segmenter = SegmentationModule(
                            depth_threshold=2.5,
                            grid_resolution=0.1,
                            grid_size=(100, 100)
                        )
        self.sample_rate = rate


    def pipeline(self, source):
        """
        Pipeline
        1. Samples Frames 
        2. Converts Frame to Depth Map
        3. Segments Depth Map
        4. Converts Segmented Data into Spatial Audio 
        """

        # Debug Print
        log(f"Running MDE -> Spatial Audio Pipeline...")

        # Open Video Path
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"Error: Could not reach video signal {source}")
        
        # Establish Frame Count
        frame_cnt = 0

        # Core Pipeline Loop
        while True:
            # Read Frame While Possible
            ret, frame = cap.read()
            if not ret:
                break

            # Sample and Process Frame in multiples of "Sample Rate". 
            if frame_cnt % self.sample_rate == 0:
                self.process(frame)

            # Increment Frame Count
            frame_cnt += 1
            
        # Release Video Source
        cap.release()

        log("Complete.")

        return NotImplemented


    def process(self, frame):
        """
        Process
        Denotes one singular pass in the pipeline.
        1. Captures Depth Map
        2. Segments Depth Map and creates BEV.
        3. Synthesizes into Spatial Audio.
        
        """

        # Preprocess
        pil_img = Image.fromarray(frame)

        # MDE Estimation
        mapping = self.model.infer_depth(pil_img)

        # Segment
        seg_results = self.segmenter.process_depth_map(mapping)

def log(msg):
    if DEBUG:
        print(msg)