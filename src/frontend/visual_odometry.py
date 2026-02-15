import logging
import numpy as np
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
                 logging: bool = False
                 ):
        self.extractor = extractor()
        self.matcher = matcher()

    def _apply_feature_extraction(self, frame1, frame2):
        return self.extractor.detect_and_compute(frame1, frame2)

    def _apply_feature_matching(self, descriptor1, descriptor2):
        return self.matcher.match(descriptor1, descriptor2)

    def run_frontend_pipeline(self, frame_pair: tuple[np.array, np.array], ):
        # feature extraction
        keypoints_frame1, descriptors_frame1, keypoints_frame2, descriptors_frame2 = self._apply_feature_extraction(frame_pair[0], frame_pair[1])

        # feature matching
        matches = self._apply_feature_matching(descriptors_frame1, descriptors_frame2)

        # Extract matched keypoints
        # pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        # pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        
        # # Camera intrinsic parameters (example values, replace with your camera's calibration data)
        # K = np.array([[fx, 0, cx],
        #             [0, fy, cy],
        #             [0, 0, 1]])
        
        # # Compute the Fundamental matrix using RANSAC
        # F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        
        # # Compute the Essential matrix using the camera's intrinsic parameters 
        # E = K.T @ F @ K
        
        # # Decompose the Essential matrix to get R and t
        # _, R, t, mask = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)

        return matches, keypoints_frame1, keypoints_frame2
    
