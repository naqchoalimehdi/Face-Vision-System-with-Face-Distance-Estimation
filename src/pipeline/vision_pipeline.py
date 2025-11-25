"""Main vision pipeline integrating all components."""

import cv2
import numpy as np
import time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from ..detection.face_detector import FaceDetector, FaceDetection
from ..detection.landmark_detector import LandmarkDetector, FaceLandmarks
from ..tracking.multi_tracker import MultiTracker, Track
from ..distance.distance_estimator import DistanceEstimator
from ..calibration.camera_calibrator import CameraCalibration
from ..utils.config_loader import ConfigLoader
from ..utils.visualization import Visualizer


@dataclass
class PipelineResult:
    """Vision pipeline result for a single frame."""
    frame: np.ndarray
    detections: List[FaceDetection]
    tracks: List[Track]
    landmarks: List[Optional[FaceLandmarks]]
    distances: List[Optional[float]]
    fps: float
    frame_count: int


class VisionPipeline:
    """Main vision pipeline orchestrating all components."""
    
    def __init__(
        self,
        config: Optional[ConfigLoader] = None,
        calibration: Optional[CameraCalibration] = None
    ):
        """
        Initialize vision pipeline.
        
        Args:
            config: Configuration loader
            calibration: Camera calibration
        """
        # Load config
        if config is None:
            from ..utils.config_loader import get_config
            config = get_config()
        
        self.config = config
        
        # Initialize components
        self._init_detector()
        self._init_landmark_detector()
        self._init_tracker()
        self._init_distance_estimator(calibration)
        self._init_visualizer()
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()
        self.fps_update_interval = 10  # Update FPS every N frames
    
    def _init_detector(self):
        """Initialize face detector."""
        det_config = self.config.get_section('detection')
        
        self.detector = FaceDetector(
            model_path=det_config.get('model_path', 'yolov8n.pt'),
            confidence_threshold=det_config.get('confidence_threshold', 0.5),
            iou_threshold=det_config.get('iou_threshold', 0.45),
            device=det_config.get('device', 'cuda')
        )
    
    def _init_landmark_detector(self):
        """Initialize landmark detector."""
        lm_config = self.config.get_section('landmarks')
        
        if lm_config.get('enabled', True):
            self.landmark_detector = LandmarkDetector(
                model_complexity=lm_config.get('model_complexity', 1),
                min_detection_confidence=lm_config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=lm_config.get('min_tracking_confidence', 0.5)
            )
        else:
            self.landmark_detector = None
    
    def _init_tracker(self):
        """Initialize multi-tracker."""
        track_config = self.config.get_section('tracking')
        
        self.tracker = MultiTracker(
            max_disappeared=track_config.get('max_disappeared', 30),
            max_distance=track_config.get('max_distance', 50)
        )
    
    def _init_distance_estimator(self, calibration: Optional[CameraCalibration]):
        """Initialize distance estimator."""
        dist_config = self.config.get_section('distance')
        
        self.distance_estimator = DistanceEstimator(
            method=dist_config.get('method', 'facial_width'),
            average_face_width_cm=dist_config.get('average_face_width_cm', 14.5),
            calibration=calibration
        )
    
    def _init_visualizer(self):
        """Initialize visualizer."""
        vis_config = self.config.get_section('visualization')
        
        self.visualizer = Visualizer(
            bbox_color=tuple(vis_config.get('bbox_color', [0, 255, 0])),
            landmark_color=tuple(vis_config.get('landmark_color', [255, 0, 0])),
            text_color=tuple(vis_config.get('text_color', [255, 255, 255])),
            thickness=vis_config.get('thickness', 2),
            font_scale=vis_config.get('font_scale', 0.6)
        )
        
        self.vis_config = vis_config
    
    def process_frame(
        self,
        frame: np.ndarray,
        visualize: bool = True
    ) -> PipelineResult:
        """
        Process single frame through pipeline.
        
        Args:
            frame: Input frame (BGR)
            visualize: Whether to draw visualizations
            
        Returns:
            PipelineResult object
        """
        # Detect faces
        detections, _ = self.detector.detect(frame)
        
        # Update tracker
        detection_data = [(det.bbox, det.confidence) for det in detections]
        tracks = self.tracker.update(detection_data)
        
        # Detect landmarks
        landmarks = []
        if self.landmark_detector is not None:
            track_bboxes = [track.bbox for track in tracks]
            landmarks = self.landmark_detector.detect_multiple(frame, track_bboxes)
        
        # Estimate distances
        distances = []
        if len(landmarks) > 0:
            distances = self.distance_estimator.estimate_multiple(
                landmarks,
                [track.bbox for track in tracks]
            )
        
        # Update FPS
        self.frame_count += 1
        if self.frame_count % self.fps_update_interval == 0:
            current_time = time.time()
            elapsed = current_time - self.last_time
            self.fps = self.fps_update_interval / elapsed
            self.last_time = current_time
        
        # Visualize
        output_frame = frame.copy()
        if visualize:
            output_frame = self.visualize(
                frame,
                tracks,
                landmarks,
                distances
            )
        
        return PipelineResult(
            frame=output_frame,
            detections=detections,
            tracks=tracks,
            landmarks=landmarks,
            distances=distances,
            fps=self.fps,
            frame_count=self.frame_count
        )
    
    def visualize(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        landmarks: List[Optional[FaceLandmarks]],
        distances: List[Optional[float]]
    ) -> np.ndarray:
        """
        Visualize results on frame.
        
        Args:
            frame: Input frame
            tracks: List of tracks
            landmarks: List of landmarks
            distances: List of distances
            
        Returns:
            Annotated frame
        """
        img = frame.copy()
        
        # Draw tracks
        if self.vis_config.get('show_bbox', True):
            img = self.visualizer.draw_tracks(
                img,
                tracks,
                distances if self.vis_config.get('show_distance', True) else None,
                self.vis_config.get('show_track_id', True)
            )
        
        # Draw landmarks
        if self.vis_config.get('show_landmarks', True) and landmarks:
            for lm in landmarks:
                if lm is not None:
                    img = self.visualizer.draw_landmarks(img, lm, draw_all=False)
        
        # Draw FPS
        if self.vis_config.get('show_fps', True):
            img = self.visualizer.draw_fps(img, self.fps)
        
        return img
    
    def process_video(
        self,
        video_source: any,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        display: bool = True
    ) -> List[PipelineResult]:
        """
        Process video stream.
        
        Args:
            video_source: Video source (file path, camera index, or VideoCapture)
            output_path: Optional output video path
            max_frames: Maximum frames to process
            display: Whether to display results
            
        Returns:
            List of PipelineResult objects
        """
        # Open video source
        if isinstance(video_source, cv2.VideoCapture):
            cap = video_source
        else:
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {video_source}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self.process_frame(frame, visualize=True)
                results.append(result)
                
                # Write frame
                if writer is not None:
                    writer.write(result.frame)
                
                # Display frame
                if display:
                    cv2.imshow('Vision Pipeline', result.frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    break
        
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        return results
    
    def set_calibration(self, calibration: CameraCalibration):
        """Update camera calibration."""
        self.distance_estimator.set_calibration(calibration)
    
    def update_config(self, updates: Dict):
        """Update configuration."""
        self.config.update(updates)
        
        # Reinitialize affected components
        if 'detection' in updates:
            self._init_detector()
        if 'landmarks' in updates:
            self._init_landmark_detector()
        if 'tracking' in updates:
            self._init_tracker()
        if 'distance' in updates:
            self._init_distance_estimator(self.distance_estimator.calibration)
        if 'visualization' in updates:
            self._init_visualizer()
    
    def reset(self):
        """Reset pipeline state."""
        self.tracker.reset()
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()
    
    def __repr__(self) -> str:
        return (
            f"VisionPipeline(detector={self.detector}, "
            f"tracker={self.tracker}, "
            f"fps={self.fps:.1f})"
        )
