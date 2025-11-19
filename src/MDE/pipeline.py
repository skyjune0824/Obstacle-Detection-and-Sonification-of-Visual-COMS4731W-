# Video Processing
import cv2

# Modules
from MDE.estimator import MDE
from src.Segmentation.segmentation import SegmentationModule

def monocular_depth_estimation_pipeline(source, debug: bool):
    """
    The following function marks the pipeline for estimating a depth map,
    performing segmentation, and synthesizing spatial audio. 

    Source: Path to video we desire to perform our pipeline on.
    Debug: Toggles Verbose
    
    """

    # Debug Print
    log(f"Performing MDE on video located at: {source}")

    # Open Video Path
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise ValueError(f"Error: Could not reach video signal {source}")
    
    # Initialize MDE Model
    model = MDE()

    # Initialize Segmentation Module
    segmenter = SegmentationModule(
        depth_threshold=2.5,
        grid_resolution=0.1,
        grid_size=(100, 100)
    )

    # Establish Sampling Rate and Frame Count
    sampling_rate = 4
    frame_cnt = 0

    # Core Pipeline Loop
    while True:
        log(f"Running MDE -> Spatial Audio Pipeline...")

        # Read frame While Possible
        ret, frame = cap.read()
        if not ret:
            break

        # Sample and Process Frame In multiples of "Sample Rate". 
        if frame_cnt % sampling_rate == 0:
            # MDE Estimation
            mapping = model.infer_depth(frame)

            # Segment
            seg_results = segmenter.process_depth_map(mapping)

    return NotImplemented


def log(msg, debug: bool):
    if debug:
        print(msg)