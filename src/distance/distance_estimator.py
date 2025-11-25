"""Distance estimation using facial geometry and camera calibration."""

import cv2
import numpy as np
from typing import Optional, Tuple
from ..detection.landmark_detector import FaceLandmarks
from ..calibration.camera_calibrator import CameraCalibration


class DistanceEstimator:
    """Estimate distance to face using various methods."""
    
    def __init__(
        self,
        method: str = "facial_width",
        average_face_width_cm: float = 14.5,
        calibration: Optional[CameraCalibration] = None
    ):
        """
        Initialize distance estimator.
        
        Args:
            method: Estimation method ('facial_width' or 'pose_estimation')
            average_face_width_cm: Average face width in cm
            calibration: Camera calibration (required for pose estimation)
        """
        self.method = method
        self.average_face_width_cm = average_face_width_cm
        self.calibration = calibration
        
        # 3D model points for pose estimation (in cm)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -6.5, -2.0),           # Chin
            (-4.5, 3.5, -2.0),           # Left eye left corner
            (4.5, 3.5, -2.0),            # Right eye right corner
            (-3.0, -3.0, -2.0),          # Left mouth corner
            (3.0, -3.0, -2.0)            # Right mouth corner
        ], dtype=np.float64)
        
        # Landmark indices for pose estimation
        self.pose_landmark_indices = [1, 152, 33, 263, 61, 291]
    
    def estimate_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        focal_length: Optional[float] = None
    ) -> float:
        """
        Estimate distance using bounding box width.
        
        Args:
            bbox: Face bounding box (x1, y1, x2, y2)
            focal_length: Camera focal length (pixels). If None, estimated.
            
        Returns:
            Distance in centimeters
        """
        x1, y1, x2, y2 = bbox
        face_width_px = x2 - x1
        
        # Estimate focal length if not provided
        if focal_length is None:
            if self.calibration is not None:
                focal_length = self.calibration.camera_matrix[0, 0]
            else:
                # Rough estimate: focal_length ≈ image_width
                focal_length = 640  # Default assumption
        
        # Distance = (Real Width × Focal Length) / Pixel Width
        distance_cm = (
            self.average_face_width_cm * focal_length / face_width_px
        )
        
        return distance_cm
    
    def estimate_from_landmarks(
        self,
        landmarks: FaceLandmarks,
        focal_length: Optional[float] = None
    ) -> float:
        """
        Estimate distance using facial landmarks.
        
        Args:
            landmarks: FaceLandmarks object
            focal_length: Camera focal length (pixels)
            
        Returns:
            Distance in centimeters
        """
        if self.method == "facial_width":
            return self._estimate_facial_width(landmarks, focal_length)
        elif self.method == "pose_estimation":
            return self._estimate_pose(landmarks)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _estimate_facial_width(
        self,
        landmarks: FaceLandmarks,
        focal_length: Optional[float] = None
    ) -> float:
        """Estimate distance using facial width from landmarks."""
        # Get face width from ear to ear
        left_ear = landmarks.key_points['left_ear']
        right_ear = landmarks.key_points['right_ear']
        
        face_width_px = np.linalg.norm(right_ear - left_ear)
        
        # Estimate focal length if not provided
        if focal_length is None:
            if self.calibration is not None:
                focal_length = self.calibration.camera_matrix[0, 0]
            else:
                focal_length = 640
        
        # Calculate distance
        distance_cm = (
            self.average_face_width_cm * focal_length / face_width_px
        )
        
        return distance_cm
    
    def _estimate_pose(
        self,
        landmarks: FaceLandmarks
    ) -> float:
        """Estimate distance using 3D pose estimation (PnP)."""
        if self.calibration is None:
            raise ValueError("Camera calibration required for pose estimation")
        
        # Extract 2D image points
        image_points = np.array([
            landmarks.landmarks_px[idx]
            for idx in self.pose_landmark_indices
        ], dtype=np.float64)
        
        # Get camera parameters
        camera_matrix = self.calibration.camera_matrix
        dist_coeffs = self.calibration.dist_coeffs
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            # Fall back to facial width method
            return self._estimate_facial_width(landmarks)
        
        # Distance is the Z component of translation vector
        distance_cm = translation_vector[2][0]
        
        return abs(distance_cm)
    
    def estimate_multiple(
        self,
        landmarks_list: list[Optional[FaceLandmarks]],
        bboxes: Optional[list[Tuple[int, int, int, int]]] = None
    ) -> list[Optional[float]]:
        """
        Estimate distances for multiple faces.
        
        Args:
            landmarks_list: List of FaceLandmarks (None for failed detections)
            bboxes: Optional list of bounding boxes (fallback if no landmarks)
            
        Returns:
            List of distances in cm (None for failed estimations)
        """
        distances = []
        
        for i, landmarks in enumerate(landmarks_list):
            if landmarks is not None:
                try:
                    distance = self.estimate_from_landmarks(landmarks)
                    distances.append(distance)
                except Exception as e:
                    print(f"Distance estimation failed: {e}")
                    distances.append(None)
            elif bboxes is not None and i < len(bboxes):
                try:
                    distance = self.estimate_from_bbox(bboxes[i])
                    distances.append(distance)
                except Exception as e:
                    distances.append(None)
            else:
                distances.append(None)
        
        return distances
    
    def set_calibration(self, calibration: CameraCalibration):
        """Update camera calibration."""
        self.calibration = calibration
    
    def __repr__(self) -> str:
        return (
            f"DistanceEstimator(method='{self.method}', "
            f"face_width={self.average_face_width_cm}cm, "
            f"calibrated={self.calibration is not None})"
        )
