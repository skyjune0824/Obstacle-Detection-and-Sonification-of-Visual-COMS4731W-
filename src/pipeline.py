from depth_estimation.estimator import MDE
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def preprocess(frame):
    """
    Processes frame for depth estimation.

    """

    # Rotate image 90 degrees (Vertical phone image)
    image_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 

    # Convert to PIL format
    pil_img = Image.fromarray(image_rotated)   

    # Further preprocessing as needed...

    # Output file
    return pil_img

def output_map(depth_map, output_dir, name):
    # Normalize depth values
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_img = Image.fromarray((depth_norm * 255).astype(np.uint8))

    # Return.
    output_path = os.path.join(output_dir, name)
    depth_img.save(output_path)


def pipeline(video_signal=None, output_dir=None, sampling_rate=2, debug=False):
    """
    Pipeline for depth estimation from live video input.

    This function performs the following:
    - Captures video input from chosen interface
    - Preprocesses each frame
    - Performs depth estimation using the MDE model
    - Post-processes the resulting depth map
    - Sends the processed data to the next stage in the pipeline

    """

    # Capture Video per input
    if video_signal is None:
        # Default to live webcam
        video_path = 0
    elif isinstance(video_signal, int):
        # Separate camera index
        video_path = video_signal
    elif isinstance(video_signal, str):
        # For testing purposes, accepts video file path
        video_path = video_signal
    else:
        raise ValueError("Invalid video signal input.")


    # Creates output directory if needed for debugging purposes.
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Capture video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not reach video signal {video_path}")
    
    # Initialize MDE Model
    model = MDE()

    # Video Sampling
    frame_count = 0

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Debugging Stream
        if debug:
            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Core Pipeline Steps
        if frame_count % sampling_rate == 0:
            # Preprocess
            frame = preprocess(frame)

            # MDE Estimation
            mapping = model.infer_depth(frame)

            # Output Debugging
            if debug and output_dir is not None:
                output_map(mapping, output_dir, f"map_{frame_count:05d}.jpg")
        
        frame_count += 1

    cap.release()
    print(f"Finished Processing.")

