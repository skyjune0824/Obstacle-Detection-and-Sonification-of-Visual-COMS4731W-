# Debug Setting
from config import DEBUG

# Video Processing
import cv2

# Image Processing
from PIL import Image
import numpy as np

# Modules
from src.MDE.estimator import MDE
from old.depth2visualize import depth2ad
from src.Segmentation.segmentation import SegmentationModule

class MDE_Pipeline:
    """
    Monocular Depth Estimation Pipeline

    Processes a video into spatial audio for obstacle detection via monocular depth estimation,
    segmentation, and spatial audio synthesis.
    """

    def __init__(self, rate, threshold):
        self.model = MDE()
        self.segmenter = SegmentationModule(
                            depth_threshold=threshold,
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

        # Debugging Visualization
        if DEBUG:
            self.visualize_depth(mapping, seg_results, frame)


    def visualize_depth(self, depth_map, seg_results, frame):
        """
        Visualize Depth

        Used for debugging depth map output of sampled frame.
        """
        vis, grid_vis = self.segmenter.visualize_results(depth_map, seg_results, frame)
        cv2.imshow('Navigation View', vis)
        cv2.imshow('Occupancy Grid', grid_vis)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        # # Normalize depth values
        # depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # depth_img = Image.fromarray((depth_norm * 255).astype(np.uint8))

        # # Display Depth Map
        # depth_img.show()

def log(msg):
    if DEBUG:
        print(msg)
