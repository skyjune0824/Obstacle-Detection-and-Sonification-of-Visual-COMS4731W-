import depth_estimation.estimator
import cv2
import os

def preprocess(frame, saved_count, output_dir):
    """
    Processes frame for depth estimation.

    """

    # Rotate image 90 degrees (Vertical phone image)
    image_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)    

    # Further preprocessing as needed...

    # Output file
    return image_rotated


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
    
    frame_count = 0
    saved_count = 0
    frame_interval = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Preprocess
            frame = preprocess(frame, saved_count, output_dir)

            # MDE Estimation


            saved_count += 1
        
        frame_count += 1

    cap.release()
    print(f"Finished sampling. Saved {saved_count} frames to '{output_dir}'.")

if __name__ == "__main__":
    # Hard coded simple paths for my own testing currently.
    input_video = "/Users/christianscaff/Documents/Academics/Columbia/Fall_25/COMS_4731/Project/Data/IMG_2200.MOV"
    output_folder = "/Users/christianscaff/Documents/Academics/Columbia/Fall_25/COMS_4731/Project"
    sample_frames(input_video, output_folder)