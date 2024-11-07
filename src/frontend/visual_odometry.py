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

    def run_frontend_pipeline(self, frame_pair: tuple[np.array, np.array]):
        # feature extraction
        keypoints_frame1, descriptors_frame1, keypoints_frame2, descriptors_frame2 = self._apply_feature_extraction(frame_pair[0], frame_pair[1])

        # feature matching
        matches = self._apply_feature_matching(descriptors_frame1, descriptors_frame2)
        # sort by distance
        matches = sorted(matches, key=lambda x: x.distance)