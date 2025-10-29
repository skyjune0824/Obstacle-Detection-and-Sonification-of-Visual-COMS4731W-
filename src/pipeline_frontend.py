from depth_estimation.estimator import MDE
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time


def preprocess(frame, saved_count):
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

def test_output(depth_map, output_dir, name):
    # Normalize
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_img = Image.fromarray((depth_norm * 255).astype(np.uint8))

    # Return for testing.
    output_path = os.path.join(output_dir, name)
    depth_img.save(output_path)


def sample_frames(video_path, output_dir):
    """
    Samples frames from video input for depth detection.
    
    """

    # Generates outout directory.
    os.makedirs(output_dir, exist_ok=True)

    # Capture video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Initialize MDE Model
    model = MDE()
    
    frame_count = 0
    saved_count = 0
    
    # Get FPS
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # if fps <= 0:
    #     fps = 30  # fallback
    # frame_interval = int(fps / 30) if fps > 30 else 1

    frame_interval = 2

    # Store Inference Times
    times = []

    # Time
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Preprocess
            frame = preprocess(frame, saved_count)

            infer_start = time.time()
            # MDE Estimation
            mapping = model.infer_depth(frame)

            infer_end = time.time()

            print(f"Inference step {saved_count} = {infer_end - infer_start} s.")

            times.append(infer_end - infer_start)

            # Return
            test_output(mapping, output_dir, f"map_{saved_count:05d}.jpg")

            saved_count += 1
        
        frame_count += 1

    end = time.time()

    cap.release()
    print(f"Finished sampling. Saved {saved_count} frames to '{output_dir}' in {end - start} s.")

    # Plot Times
    x = np.arange(saved_count)
    y = np.array(times)

    mean_value = np.mean(times)
    plt.axhline(mean_value, color='red', linestyle='--', label=f"Mean = {mean_value:.2f}s")
    
    plt.plot(x, y)

    plt.xlabel("Frame")
    plt.ylabel("Time (s)")
    plt.title("Inference Time Per Frame")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # Hard coded simple paths for my own testing currently.
    input_video = "/Users/christianscaff/Documents/Academics/Columbia/Fall_25/COMS_4731/Project/Data/IMG_2200.MOV"
    output_folder = "/Users/christianscaff/Documents/Academics/Columbia/Fall_25/COMS_4731/Project/Result"
    sample_frames(input_video, output_folder)