"""Detection modules."""

from .face_detector import FaceDetector, FaceDetection
from .landmark_detector import LandmarkDetector, FaceLandmarks

__all__ = [
    'FaceDetector',
    'FaceDetection',
    'LandmarkDetector',
    'FaceLandmarks'
]
