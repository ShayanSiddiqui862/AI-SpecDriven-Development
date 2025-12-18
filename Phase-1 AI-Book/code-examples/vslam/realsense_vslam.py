#!/usr/bin/env python3

"""
RealSense VSLAM Implementation
Hardware-accelerated Visual SLAM using Intel RealSense camera
"""

import cv2
import numpy as np
import math
import time
from collections import deque
import threading

try:
    import pyrealsense2 as rs
except ImportError:
    print("pyrealsense2 not available, using simulated camera")


class RealSenseVSLAM:
    def __init__(self):
        # Camera parameters (typical for RealSense D435)
        self.fx = 616.17  # Focal length x
        self.fy = 616.05  # Focal length y
        self.cx = 313.11  # Principal point x
        self.cy = 234.37  # Principal point y

        # Initialize ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31
        )

        # FLANN matcher for feature matching
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Pose tracking
        self.current_pose = np.eye(4)  # 4x4 identity matrix
        self.previous_pose = np.eye(4)

        # Feature tracking
        self.keypoints_prev = None
        self.descriptors_prev = None
        self.keypoints_curr = None
        self.descriptors_curr = None

        # Map points (simplified representation)
        self.map_points = []
        self.frame_id = 0

        # Camera setup
        self.pipeline = None
        self.setup_camera()

        print("RealSense VSLAM initialized")

    def setup_camera(self):
        """Setup RealSense camera or simulate camera input"""
        try:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            self.pipeline.start(config)
            print("RealSense camera connected")
        except:
            print("RealSense camera not available, using simulated input")
            self.pipeline = None

    def get_frame(self):
        """Get RGB frame from camera"""
        if self.pipeline:
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    return color_image
            except:
                pass

        # Return simulated frame if camera not available
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def detect_features(self, image):
        """Detect ORB features in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match features between two frames using FLANN"""
        if desc1 is None or desc2 is None:
            return []

        if len(desc1) < 2 or len(desc2) < 2:
            return []

        try:
            matches = self.flann.knnMatch(desc1, desc2, k=2)
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            return good_matches
        except:
            return []

    def estimate_pose(self, kp1, kp2, matches):
        """Estimate relative pose from matched features"""
        if len(matches) < 10:
            return None, None

        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find fundamental matrix
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 4, 0.999)

        if F is None or F.shape != (3, 3):
            return None, None

        # Get essential matrix
        K = np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0, 0, 1]], dtype=np.float32)

        E = K.T @ F @ K

        # Decompose essential matrix
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K)

        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()

        return T, mask

    def process_frame(self, image):
        """Process a single frame for VSLAM"""
        # Detect features
        keypoints, descriptors = self.detect_features(image)

        if self.keypoints_prev is not None and self.descriptors_prev is not None:
            # Match features with previous frame
            matches = self.match_features(self.descriptors_prev, descriptors)

            if len(matches) > 10:
                # Estimate pose
                T_rel, mask = self.estimate_pose(self.keypoints_prev, keypoints, matches)

                if T_rel is not None:
                    # Update current pose
                    self.current_pose = self.current_pose @ T_rel

                    # Add some map points (simplified)
                    if len(matches) > 50:
                        for match in matches[:20]:  # Add first 20 matches as map points
                            pt = keypoints[match.trainIdx].pt
                            depth = self.get_depth_at_point(int(pt[0]), int(pt[1]))
                            if depth > 0:
                                # Convert to 3D point
                                X = (pt[0] - self.cx) * depth / self.fx
                                Y = (pt[1] - self.cy) * depth / self.fy
                                Z = depth

                                # Transform to world coordinates
                                world_point = self.current_pose @ np.array([X, Y, Z, 1])
                                self.map_points.append(world_point[:3])

        # Update previous frame data
        self.keypoints_prev = keypoints
        self.descriptors_prev = descriptors
        self.frame_id += 1

        return self.current_pose.copy()

    def get_depth_at_point(self, x, y):
        """Get depth at specific pixel (simulated if no depth camera)"""
        if self.pipeline:
            try:
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth = depth_frame.as_depth_frame().get_distance(x, y)
                    return depth
            except:
                pass

        # Return simulated depth
        return 1.0 + 0.5 * np.sin(x * 0.01) * np.cos(y * 0.01)

    def run(self):
        """Main VSLAM loop"""
        print("Starting VSLAM loop...")

        try:
            while True:
                # Get frame
                frame = self.get_frame()

                # Process frame
                pose = self.process_frame(frame)

                # Display results
                cv2.putText(frame, f'Frame: {self.frame_id}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Pose: [{pose[0,3]:.2f}, {pose[1,3]:.2f}, {pose[2,3]:.2f}]',
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                cv2.imshow('VSLAM - Press Q to quit', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            print("VSLAM interrupted by user")
        finally:
            if self.pipeline:
                self.pipeline.stop()
            cv2.destroyAllWindows()

    def get_trajectory(self):
        """Return the estimated trajectory"""
        return self.current_pose


def main():
    vslam = RealSenseVSLAM()
    vslam.run()


if __name__ == '__main__':
    main()