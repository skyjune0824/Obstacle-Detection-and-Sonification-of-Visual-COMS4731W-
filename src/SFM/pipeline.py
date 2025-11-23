# Debug Setting
from config import DEBUG

# Video Processing
import cv2
from PIL import Image
import numpy as np

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

            # Initial Pose
            pose_init = np.eye(4)

            # Sample and Process Frame in multiples of "Sample Rate". 
            if frame_cnt % self.sample_rate == 0:
                # Calculate Pose
                new_pose, pts_one, pts_two = self.process_trajectory(frame_one, frame_two)
                curr_pose = pose_init @ new_pose

                # Triangulate
                self.triangulate(pose_init, curr_pose, pts_one, pts_two)


            # Increment Frame Count
            frame_cnt += 1
            
        # Release Video Source
        cap.release()

        log("Complete.")

        # return self.audio_trace
    
    def process_trajectory(self, frame_one, frame_two):
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

        # Match Points
        pts_one = np.float32([keypoint_one[m.queryIdx].pt for m in matches])
        pts_two = np.float32([keypoint_two[m.trainIdx].pt for m in matches])
        
        # Approximated Intrinsic for iPhone 14
        K = np.array([[780, 0, 540],
              [0, 780, 960],
              [0, 0, 1]])
        
        # Fundamental Matrix
        F, inliers = cv2.findFundamentalMat(pts_one, pts_two, cv2.FM_RANSAC)

        # Essential Matrix
        E = K.T @ F @ K

        # Get Points for Inliner Matches
        pts_one_inliers = pts_one[inliers.ravel() == 1]
        pts2_two_inliers = pts_two[inliers.ravel() == 1]

        # Decompose for Rotation + Translation
        _, R, t, mask = cv2.recoverPose(E, pts_one_inliers, pts2_two_inliers, K)

        # Construct Pose Matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.ravel()

        return pose, pts_one_inliers, pts2_two_inliers


    def triangulate(self, pose1, pose2, pts1, pts2):
        # Initialize array to store homogeneous coordinates of 3D points
        points_3d_h = np.zeros((pts1.shape[0], 4))
    
        # Invert camera poses to convert World Points to Camera Points
        pose1 = np.linalg.inv(pose1)
        pose2 = np.linalg.inv(pose2)
    
        # Loop through each pair of corresponding points
        for i, p in enumerate(zip(np.hstack([pts1, np.ones((pts1.shape[0], 1))]), np.hstack([pts2, np.ones((pts2.shape[0], 1))]))):
            # Initialize matrix A w/ linear equations: X = PX
            A = np.zeros((4, 4))
    
            # Populate the matrix A with equations derived from projection matrices + pts
            A[0] = p[0][0] * pose1[2] - pose1[0]
            A[1] = p[0][1] * pose1[2] - pose1[1]
            A[2] = p[1][0] * pose2[2] - pose2[0]
            A[3] = p[1][1] * pose2[2] - pose2[1]
    
            # SVD on A
            _, _, vt = np.linalg.svd(A)
    
            # Last row of (V^T) - smallest singular value = Triangulated Solution
            points_3d_h[i] = vt[3]

        # Divide Out W from [X, Y, Z, W] and remove invalid points.
        valid_filter = (np.abs(points_3d_h[:, 3]) > 0.005) & (points_3d_h[:, 2] > 0)
        pts3d_h_filtered = points_3d_h[valid_filter]  
        pts3d = pts3d_h_filtered[:, :3] / pts3d_h_filtered[:, 3][:, np.newaxis]
    
        # Return the 3D points
        log(f"Triangulated Points: {pts3d}")

        return pts3d
 
def log(msg):
    if DEBUG:
        print(msg)