"""MediaPipe-based facial landmark detection."""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FaceLandmarks:
    """Facial landmarks result."""
    landmarks: np.ndarray  # Shape: (468, 3) for normalized coordinates
    landmarks_px: np.ndarray  # Shape: (468, 2) for pixel coordinates
    bbox: Tuple[int, int, int, int]  # Bounding box (x1, y1, x2, y2)
    key_points: Dict[str, np.ndarray]  # Key facial points


class LandmarkDetector:
    """MediaPipe-based facial landmark detector."""
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False
    ):
        """
        Initialize landmark detector.
        
        Args:
            model_complexity: Model complexity (0, 1, or 2)
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
            static_image_mode: Whether to treat images as static
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define key landmark indices
        self.key_indices = {
            'left_eye': [33, 133, 160, 159, 158, 157, 173],
            'right_eye': [362, 263, 387, 386, 385, 384, 398],
            'nose_tip': [1],
            'nose_bridge': [6],
            'mouth_left': [61],
            'mouth_right': [291],
            'mouth_center': [13],
            'left_ear': [234],
            'right_ear': [454],
            'chin': [152],
            'forehead': [10]
        }
    
    def detect(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[FaceLandmarks]:
        """
        Detect facial landmarks in image.
        
        Args:
            image: Input image (BGR format)
            face_bbox: Optional face bounding box to crop region
            
        Returns:
            FaceLandmarks object or None if no face detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Optionally crop to face region
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            # Add padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            face_region = image_rgb[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            face_region = image_rgb
            offset_x, offset_y = 0, 0
        
        # Process image
        results = self.face_mesh.process(face_region)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract normalized coordinates
        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in face_landmarks.landmark
        ])
        
        # Convert to pixel coordinates
        region_h, region_w = face_region.shape[:2]
        landmarks_px = np.array([
            [
                int(lm.x * region_w) + offset_x,
                int(lm.y * region_h) + offset_y
            ]
            for lm in face_landmarks.landmark
        ])
        
        # Calculate bounding box from landmarks
        x_coords = landmarks_px[:, 0]
        y_coords = landmarks_px[:, 1]
        bbox = (
            int(x_coords.min()),
            int(y_coords.min()),
            int(x_coords.max()),
            int(y_coords.max())
        )
        
        # Extract key points
        key_points = self._extract_key_points(landmarks_px)
        
        return FaceLandmarks(
            landmarks=landmarks,
            landmarks_px=landmarks_px,
            bbox=bbox,
            key_points=key_points
        )
    
    def detect_multiple(
        self,
        image: np.ndarray,
        face_bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Optional[FaceLandmarks]]:
        """
        Detect landmarks for multiple faces.
        
        Args:
            image: Input image (BGR format)
            face_bboxes: List of face bounding boxes
            
        Returns:
            List of FaceLandmarks (None for faces where detection failed)
        """
        results = []
        for bbox in face_bboxes:
            landmarks = self.detect(image, bbox)
            results.append(landmarks)
        
        return results
    
    def _extract_key_points(
        self,
        landmarks_px: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract key facial points from landmarks.
        
        Args:
            landmarks_px: Pixel coordinates of all landmarks
            
        Returns:
            Dictionary of key points
        """
        key_points = {}
        
        for name, indices in self.key_indices.items():
            if len(indices) == 1:
                # Single point
                key_points[name] = landmarks_px[indices[0]]
            else:
                # Average of multiple points
                points = landmarks_px[indices]
                key_points[name] = points.mean(axis=0).astype(int)
        
        return key_points
    
    def get_face_width(self, landmarks: FaceLandmarks) -> float:
        """
        Calculate face width from landmarks.
        
        Args:
            landmarks: FaceLandmarks object
            
        Returns:
            Face width in pixels
        """
        left_point = landmarks.key_points['left_ear']
        right_point = landmarks.key_points['right_ear']
        
        width = np.linalg.norm(right_point - left_point)
        return float(width)
    
    def get_face_height(self, landmarks: FaceLandmarks) -> float:
        """
        Calculate face height from landmarks.
        
        Args:
            landmarks: FaceLandmarks object
            
        Returns:
            Face height in pixels
        """
        top_point = landmarks.key_points['forehead']
        bottom_point = landmarks.key_points['chin']
        
        height = np.linalg.norm(bottom_point - top_point)
        return float(height)
    
    def close(self):
        """Release resources."""
        self.face_mesh.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass
    
    def __repr__(self) -> str:
        return (
            f"LandmarkDetector(complexity={self.model_complexity}, "
            f"det_conf={self.min_detection_confidence}, "
            f"track_conf={self.min_tracking_confidence})"
        )
