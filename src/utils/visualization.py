"""Visualization utilities for drawing detections and tracking results."""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from ..detection.face_detector import FaceDetection
from ..detection.landmark_detector import FaceLandmarks
from ..tracking.multi_tracker import Track


class Visualizer:
    """Visualization utilities for face detection system."""
    
    def __init__(
        self,
        bbox_color: Tuple[int, int, int] = (0, 255, 0),
        landmark_color: Tuple[int, int, int] = (255, 0, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        font_scale: float = 0.6
    ):
        """
        Initialize visualizer.
        
        Args:
            bbox_color: Bounding box color (BGR)
            landmark_color: Landmark color (BGR)
            text_color: Text color (BGR)
            thickness: Line thickness
            font_scale: Font scale
        """
        self.bbox_color = bbox_color
        self.landmark_color = landmark_color
        self.text_color = text_color
        self.thickness = thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Color palette for different track IDs
        self.track_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
        ]
    
    def draw_detection(
        self,
        image: np.ndarray,
        detection: FaceDetection,
        label: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw single face detection.
        
        Args:
            image: Input image
            detection: FaceDetection object
            label: Optional label text
            
        Returns:
            Image with drawn detection
        """
        img = image.copy()
        x1, y1, x2, y2 = detection.bbox
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), self.bbox_color, self.thickness)
        
        # Draw label
        if label is None:
            label = f"{detection.confidence:.2f}"
        
        self._draw_label(img, label, (x1, y1 - 10))
        
        return img
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[FaceDetection]
    ) -> np.ndarray:
        """
        Draw multiple face detections.
        
        Args:
            image: Input image
            detections: List of FaceDetection objects
            
        Returns:
            Image with drawn detections
        """
        img = image.copy()
        
        for detection in detections:
            img = self.draw_detection(img, detection)
        
        return img
    
    def draw_track(
        self,
        image: np.ndarray,
        track: Track,
        distance: Optional[float] = None,
        show_id: bool = True
    ) -> np.ndarray:
        """
        Draw tracked face.
        
        Args:
            image: Input image
            track: Track object
            distance: Optional distance in cm
            show_id: Whether to show track ID
            
        Returns:
            Image with drawn track
        """
        img = image.copy()
        x1, y1, x2, y2 = track.bbox
        
        # Get color for this track
        color = self.track_colors[track.track_id % len(self.track_colors)]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, self.thickness)
        
        # Build label
        label_parts = []
        if show_id:
            label_parts.append(f"ID:{track.track_id}")
        if distance is not None:
            label_parts.append(f"{distance:.1f}cm")
        
        label = " | ".join(label_parts) if label_parts else ""
        
        if label:
            self._draw_label(img, label, (x1, y1 - 10), color)
        
        return img
    
    def draw_tracks(
        self,
        image: np.ndarray,
        tracks: List[Track],
        distances: Optional[List[Optional[float]]] = None,
        show_ids: bool = True
    ) -> np.ndarray:
        """
        Draw multiple tracked faces.
        
        Args:
            image: Input image
            tracks: List of Track objects
            distances: Optional list of distances in cm
            show_ids: Whether to show track IDs
            
        Returns:
            Image with drawn tracks
        """
        img = image.copy()
        
        for i, track in enumerate(tracks):
            distance = None
            if distances is not None and i < len(distances):
                distance = distances[i]
            
            img = self.draw_track(img, track, distance, show_ids)
        
        return img
    
    def draw_landmarks(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        draw_all: bool = False
    ) -> np.ndarray:
        """
        Draw facial landmarks.
        
        Args:
            image: Input image
            landmarks: FaceLandmarks object
            draw_all: Whether to draw all 468 points or just key points
            
        Returns:
            Image with drawn landmarks
        """
        img = image.copy()
        
        if draw_all:
            # Draw all landmarks
            for point in landmarks.landmarks_px:
                cv2.circle(
                    img,
                    tuple(point.astype(int)),
                    1,
                    self.landmark_color,
                    -1
                )
        else:
            # Draw only key points
            for name, point in landmarks.key_points.items():
                cv2.circle(
                    img,
                    tuple(point.astype(int)),
                    3,
                    self.landmark_color,
                    -1
                )
        
        return img
    
    def draw_fps(
        self,
        image: np.ndarray,
        fps: float,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        Draw FPS counter.
        
        Args:
            image: Input image
            fps: FPS value
            position: Text position (x, y)
            
        Returns:
            Image with FPS counter
        """
        img = image.copy()
        
        text = f"FPS: {fps:.1f}"
        
        # Draw background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            self.font,
            self.font_scale,
            self.thickness
        )
        
        x, y = position
        cv2.rectangle(
            img,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            img,
            text,
            position,
            self.font,
            self.font_scale,
            self.text_color,
            self.thickness
        )
        
        return img
    
    def draw_info_panel(
        self,
        image: np.ndarray,
        info: Dict[str, any],
        position: Tuple[int, int] = (10, 60)
    ) -> np.ndarray:
        """
        Draw information panel.
        
        Args:
            image: Input image
            info: Dictionary of information to display
            position: Starting position (x, y)
            
        Returns:
            Image with info panel
        """
        img = image.copy()
        x, y = position
        line_height = 25
        
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(
                img,
                text,
                (x, y),
                self.font,
                self.font_scale * 0.8,
                self.text_color,
                self.thickness - 1
            )
            y += line_height
        
        return img
    
    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        bg_color: Optional[Tuple[int, int, int]] = None
    ):
        """Draw text label with background."""
        if bg_color is None:
            bg_color = (0, 0, 0)
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            self.font,
            self.font_scale,
            self.thickness
        )
        
        x, y = position
        
        # Ensure label is within image bounds
        y = max(text_height + 5, y)
        
        # Draw background
        cv2.rectangle(
            image,
            (x, y - text_height - 5),
            (x + text_width + 5, y + baseline),
            bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image,
            text,
            (x, y),
            self.font,
            self.font_scale,
            self.text_color,
            self.thickness
        )
    
    def create_grid(
        self,
        images: List[np.ndarray],
        grid_size: Optional[Tuple[int, int]] = None,
        labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Create grid of images.
        
        Args:
            images: List of images
            grid_size: Grid size (rows, cols). If None, auto-calculated.
            labels: Optional labels for each image
            
        Returns:
            Grid image
        """
        if len(images) == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Calculate grid size if not provided
        if grid_size is None:
            n = len(images)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            grid_size = (rows, cols)
        
        rows, cols = grid_size
        
        # Get image size (assume all same size)
        h, w = images[0].shape[:2]
        
        # Create grid
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        
        for i, img in enumerate(images):
            if i >= rows * cols:
                break
            
            row = i // cols
            col = i % cols
            
            # Add label if provided
            if labels is not None and i < len(labels):
                img = img.copy()
                self._draw_label(img, labels[i], (10, 30))
            
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        return grid
