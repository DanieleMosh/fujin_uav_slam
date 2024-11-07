import cv2
from abc import ABC, abstractmethod

# abstract base class for feature extraction methods
class FeatureExtractor(ABC):
    """
    Parameters:
    - Image 1 (first in image seq pair)
    - Image 2 (second in image seq pair)

    Returns:
    - keypoints for first image
    - keypoints for second image
    - descriptors for first image
    - descriptors for second image
    """
    @abstractmethod
    def detect_and_compute(self, frame1, frame2):
        pass

# feature extractor using ORB
class ORBFeatureExtractor(FeatureExtractor):

    def __init__(self):
        self.orb = cv2.ORB_create()

    def detect_and_compute(self, frame1, frame2):
        keypoints_frame1, descriptors_frame1 = self.orb.detectAndCompute(frame1, None)
        keypoints_frame2, descriptors_frame2 = self.orb.detectAndCompute(frame2, None)

        return keypoints_frame1, descriptors_frame1, keypoints_frame2, descriptors_frame2
    
# feature extractor using SIFT
class SIFTFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def detect_and_compute(self, frame1, frame2):
        keypoints_frame1, descriptors_frame1 = self.sift.detectAndCompute(frame1, None)
        keypoints_frame2, descriptors_frame2 = self.sift.detectAndCompute(frame2, None)
        
        return keypoints_frame1, descriptors_frame1, keypoints_frame2, descriptors_frame2
    