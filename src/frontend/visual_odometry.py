import logging
import numpy as np
import cv2
from typing import Type

from frontend.feature_matching import FeatureMatcher, BFMatcher, FLANNMatcher
from frontend.feature_extraction import FeatureExtractor, ORBFeatureExtractor, SIFTFeatureExtractor

# logging.basicConfig(level=logging.INFO)

"""
Input:
    - Image Sequence (numpy arrays)

Visual Odometry flow chart:
    - Feature Extraction
    - Feature Matching
    - Motion Estimation
    - Local Optimization
"""

def add_ones(pts):
    """Helper function to add a column of ones to a 2D array (homogeneous coordinates)."""
    return np.hstack([pts, np.ones((pts.shape[0], 1))])
 
def triangulate(pose1, pose2, pts1, pts2):
    # Initialize the result array to store the homogeneous coordinates of the 3D points
    ret = np.zeros((pts1.shape[0], 4))
 
    # Invert the camera poses to get the projection matrices
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
 
    # Loop through each pair of corresponding points
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        # Initialize the matrix A to hold the linear equations
        A = np.zeros((4, 4))
 
        # Populate the matrix A with the equations derived from the projection matrices and the points
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
 
        # Perform SVD on A
        _, _, vt = np.linalg.svd(A)
 
        # The solution is the last row of V transposed (V^T), corresponding to the smallest singular value
        ret[i] = vt[3]
 
    # Return the 3D points in homogeneous coordinates
    return ret

class VisualOdometry:
    def __init__(self,
                 extractor: Type[FeatureExtractor]=ORBFeatureExtractor,
                 matcher: Type[FeatureMatcher]=BFMatcher,
                 logging: bool = False,
                 render_queue=None
                 ):
        self.extractor = extractor()
        self.matcher = matcher()

        # EuRoC cam0 intrinsics
        self.K = np.array([
            [458.654, 0.0,     367.215],
            [0.0,     457.296, 248.375],
            [0.0,     0.0,     1.0]
        ])
        self.K_inv = np.linalg.inv(self.K)

        self.current_pose = np.eye(4)
        self.render_queue = render_queue

    def _apply_feature_extraction(self, frame1, frame2):
        return self.extractor.detect_and_compute(frame1, frame2)

    def _apply_feature_matching(self, descriptor1, descriptor2):
        return self.matcher.match(descriptor1, descriptor2)

    def _estimate_motion(self, matches, kp1, kp2):
        if len(matches) < 8:
            return None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        F, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        if F is None or inlier_mask is None:
            return None

        inlier_mask = inlier_mask.ravel().astype(bool)
        pts1_inlier = pts1[inlier_mask]
        pts2_inlier = pts2[inlier_mask]

        if len(pts1_inlier) < 8:
            return None

        E = self.K.T @ F @ self.K
        _, R, t, _ = cv2.recoverPose(E, pts1_inlier, pts2_inlier, self.K)

        return R, t, pts1_inlier, pts2_inlier

    def _triangulate_points(self, R, t, pts1_inlier, pts2_inlier):
        # Build relative pose (frame2 w.r.t. frame1)
        relative_pose = np.eye(4)
        relative_pose[:3, :3] = R
        relative_pose[:3, 3] = t.ravel()

        # Normalize pixel coords to camera coords
        pts1_norm = (self.K_inv @ add_ones(pts1_inlier).T).T[:, :2]
        pts2_norm = (self.K_inv @ add_ones(pts2_inlier).T).T[:, :2]

        # Triangulate using existing SVD-based function
        pose1 = np.eye(4)
        pts4d = triangulate(pose1, relative_pose, pts1_norm, pts2_norm)

        # Dehomogenize
        pts4d /= pts4d[:, 3:]

        # Filter: keep points in front of camera and not at infinity
        good = (pts4d[:, 2] > 0) & (np.abs(pts4d[:, 3]) > 1e-6)
        pts3d = pts4d[good, :3]

        # Transform to world coordinates
        world_pts = (self.current_pose @ np.hstack([pts3d, np.ones((len(pts3d), 1))]).T).T[:, :3]

        return world_pts

    def run_frontend_pipeline(self, frame_pair: tuple[np.array, np.array]):
        # feature extraction
        keypoints_frame1, descriptors_frame1, keypoints_frame2, descriptors_frame2 = self._apply_feature_extraction(frame_pair[0], frame_pair[1])

        # feature matching
        matches = self._apply_feature_matching(descriptors_frame1, descriptors_frame2)

        # motion estimation
        result = self._estimate_motion(matches, keypoints_frame1, keypoints_frame2)
        if result is not None:
            R, t, pts1_inlier, pts2_inlier = result

            # triangulate 3D points
            world_pts = self._triangulate_points(R, t, pts1_inlier, pts2_inlier)

            # accumulate pose: current_pose = current_pose @ inv(relative)
            relative_pose = np.eye(4)
            relative_pose[:3, :3] = R
            relative_pose[:3, 3] = t.ravel()
            self.current_pose = self.current_pose @ np.linalg.inv(relative_pose)

            # push to renderer
            if self.render_queue is not None:
                try:
                    self.render_queue.put_nowait({
                        'pose': self.current_pose.copy(),
                        'points': world_pts
                    })
                except Exception:
                    pass  # queue full, drop frame

        return matches, keypoints_frame1, keypoints_frame2
    
