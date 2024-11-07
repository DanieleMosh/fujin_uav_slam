import cv2
import numpy as np
from abc import ABC, abstractmethod

# abstract base class for feature matching methods
# match descriptors between two images

class FeatureMatcher(ABC):
    """
    Parameters:
    - descriptor1
    - descriptor2

    Returns:
    - matches
    """
    @abstractmethod
    def match(self, descriptors1, descriptors2):
        pass

class BFMatcher(FeatureMatcher):
    def __init__(self, cross_check=True, norm_type=cv2.NORM_L2):
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=cross_check)

    def match(self, descriptors1, descriptors2):
        matches = self.matcher.match(descriptors1, descriptors2)
        return sorted(matches, key=lambda x: x.distance)

class FLANNMatcher(FeatureMatcher):
    def __init__(self):
        # FLANN parameters for SIFT and SURF
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, descriptors1, descriptors2):
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        # apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches