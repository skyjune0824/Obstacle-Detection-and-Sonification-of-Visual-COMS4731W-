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

    def __init__(self, rate, K):
        self.synth = SpatialAudioFeedback()
        self.sample_rate = rate
        self.cap = None
        self.K = K
        self.world_map = []


    def pipeline(self, source):
        """
        The following function marks the pipeline for creating a minimal point cloud
        from multiple images in motion, performing segmentation, and synthesizing spatial audio.

        Source: Path to video we desire to perform our pipeline on.
        Debug: Toggles Verbose

        """

        print(f"Performing SFM on video located at: {source}")

        # Open Video Path
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not reach video signal {source}")
        
        # Establish Frame Count
        frame_cnt = 0

        # Initial Pose
        prev_pose = np.eye(4)
        prev_frame = None

        # Core Pipeline Loop
        while True:
            # Read Frame While Possible
            ret, cur_frame = self.cap.read()
            if not ret:
                break

            # Sample and Process Frame in multiples of "Sample Rate". 
            if prev_frame is not None:
                # Calculate Pose
                relative_pose, pts_one, pts_two = self.process_trajectory(prev_frame, cur_frame)

                # Update World Pose
                curr_pose = prev_pose @ relative_pose

                # Triangulate
                if pts_one.any() and pts_two.any():
                    local_pose1 = np.eye(4)
                    local_points = self.triangulate(local_pose1, relative_pose, pts_one, pts_two)

                    # Convert Local Points to World Points
                    world_points_current = self.transform_to_world(local_points, prev_pose)

                    # Add to map
                    if world_points_current.any():
                        self.world_map.extend(world_points_current)
                        log(f"World Map Added {len(world_points_current)}. TOTAL SIZE: {len(self.world_map)}")

                    # Find Distance to closest object in each zone.
                    minimum, min_pnt = self.min_distance(curr_pose[:3, 3])

                    # Visualize
                    if DEBUG and min_pnt is not None and local_points.any():
                        self.viz_dist_points(min_pnt, minimum, curr_pose, cur_frame)

                # Set Prev Pose
                prev_pose = curr_pose

            # Increment Frame Count
            prev_frame = cur_frame.copy()
            frame_cnt += 1
            
        # Release Video Source
        self.cap.release()

        log("Complete.")

        # return self.audio_trace
    
    def process_trajectory(self, frame_one, frame_two):
        """ Process
        Denotes one singular pass in the pipeline.
        1. Captures Depth Map
        2. Segments Depth Map and creates BEV.
        3. Synthesizes into Spatial Audio.
        
        """

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

        # Match Points
        pts_one = np.float32([keypoint_one[m.queryIdx].pt for m in matches])
        pts_two = np.float32([keypoint_two[m.trainIdx].pt for m in matches])
        
        # Fundamental Matrix
        F, inliers = cv2.findFundamentalMat(pts_one, pts_two, cv2.FM_RANSAC)

        # Essential Matrix
        E = self.K.T @ F @ self.K

        # Get Points for Inliner Matches
        pts_one_inliers = pts_one[inliers.ravel() == 1]
        pts_two_inliers = pts_two[inliers.ravel() == 1]

        # Decompose for Rotation + Translation
        _, R, t, mask = cv2.recoverPose(E, pts_one_inliers, pts_two_inliers, self.K)

        # Mask
        pts_one_masked = pts_one_inliers[mask.ravel() > 0]
        pts_two_masked = pts_two_inliers[mask.ravel() > 0]

        # Normalize Inliners
        k_inv = np.linalg.inv(self.K)
        pts_one_normed = np.dot(k_inv, np.hstack([pts_one_masked, np.ones((pts_one_masked.shape[0], 1))]).T).T[:, 0:2]
        pts_two_normed = np.dot(k_inv, np.hstack([pts_two_masked, np.ones((pts_two_masked.shape[0], 1))]).T).T[:, 0:2]

        # Construct Pose Matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.ravel()

        # Scale Pose to Walking Speed
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        walking_speed = 1.4  # meters per second 
        estimated_scale_translation = walking_speed * (1 / fps)
        pose[:3, 3] *= estimated_scale_translation

        return pose, pts_one_normed, pts_two_normed


    def triangulate(self, pose1, pose2, pts1, pts2):    
        # Invert camera poses to convert World Points to Camera Points
        pose1 = pose1[:3, :]
        pose2 = pose2[:3, :]

        # DEBUG
        # if DEBUG:
        #     log(f"POSE1: {pose1},\n POSE2: {pose2}")
        #     log(f"pts1: {pts1.T}, pts2: {pts2.T}")
    
        points_4d = cv2.triangulatePoints(pose1, pose2, pts1.T, pts2.T)
        points_4d = points_4d.T

        # log(f"Before Filter: {len(points_4d)}")
        # Filter and Divide Out W from [X, Y, Z, W]
        valid_filter = (np.abs(points_4d[:, 3]) > 0.005)
        points_4d = points_4d[valid_filter]

        points_4d /= points_4d[:, 3:]
        points_3d = points_4d[:, :3]

        # Filter Depth that is too far or behind the camera (Numerical Instability)
        reasonable_depth = (points_3d[:, 2] < 80.0) & (points_3d[:, 2] > 0.1)
        points_3d = points_3d[reasonable_depth]

        # log(f"After Filter: {len(points_3d)}")

        # if DEBUG and points_3d.any():
        #     log(f"Triangulated {len(points_3d)} points")
        #     log(f"Z range: {points_3d[:, 2].min():.2f} to {points_3d[:, 2].max():.2f}")
        #     log(f"XY range: X[{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}], "
        #         f"Y[{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}]")
                
        # Return the 3D points
        return points_3d
    
    def transform_to_world(self, local_points, camera_pose):
        if len(local_points) == 0:
            return np.array([])

        # Convert to homogeneous coordinates - Add Column of Ones
        ones = np.ones((local_points.shape[0], 1))
        local_homogeneous = np.hstack([local_points, ones])

        # Transform to world coordinates 
        world_homogeneous = (camera_pose @ local_homogeneous.T).T

        # Return 3D coordinates
        return world_homogeneous[:, :3]

    def min_distance(self, cam_pos, count=100):
        if len(self.world_map) > 0:
            world_pts_array = np.array(self.world_map[-count:])

            # Filter Only Points on Front of Camera
            in_front_mask = world_pts_array[:, 2] > cam_pos[2]
            valid_pts = world_pts_array[in_front_mask]

            if valid_pts.any():
                dists = np.linalg.norm(valid_pts - cam_pos, axis=1)
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]
                min_pnt = valid_pts[min_idx]

                return min_dist, min_pnt
  
        return float('inf'), None
    

    def transform_to_local(self, camera_pose, point): 
        # Invert Pose
        camera_pose_inv = np.linalg.inv(camera_pose)
        
        # Convert point to homogeneous coordinates
        point_homogeneous = np.append(point, 1)
        
        # Transform
        camera_homogeneous = camera_pose_inv @ point_homogeneous
        
        # Return 3D
        return camera_homogeneous[:3]

    
    def viz_dist_points(self, min_pnt, min_dist, cur_pose, cur_frame):
        # Get Local Points In Frame
        cam_points = []

        for point in self.world_map[-50:]:
            cam_points.append(self.transform_to_local(cur_pose, point))

        cam_points = np.vstack(cam_points)

        # Get Min Point
        min_point = self.transform_to_local(cur_pose, min_pnt)

        # Filter Pos Z points
        in_front = cam_points[:, 2] > 0.1
        camera_points_front = cam_points[in_front]

        # All Triangulated Positions
        for pnt in camera_points_front:
            ret = np.dot(self.K, pnt)
            ret /= ret[2]

            x = int(round(ret[0]))
            y = int(round(ret[1]))
            cv2.circle(cur_frame, (x, y), 8, (0, 255, 0), -1)

        # Get Min Point
        ret = np.dot(self.K, min_point)
        ret /= ret[2]

        x = int(round(ret[0]))
        y = int(round(ret[1]))

        cv2.circle(cur_frame, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            cur_frame,
            f"{min_dist:.2f}m",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        log(f"MIN DIST: {min_dist} @ ({x}, {y})")


        cv2.imshow("Closest Point Visualization", cur_frame)
        cv2.waitKey(10)


 
def log(msg):
    if DEBUG:
        print(msg)

# Notes
# I could try and add all the points and poses to an accumulating map
# and calculate each pose relative to the world position instead of local and see 
# if that improved distance accuracy (Pull min distance from overall map)
# I could also take clusters or median distances