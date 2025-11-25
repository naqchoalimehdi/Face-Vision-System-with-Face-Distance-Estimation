"""YOLOv8-based face detection module."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO
from dataclasses import dataclass


@dataclass
class FaceDetection:
    """Face detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0  # Face class


class FaceDetector:
    """YOLOv8-based face detector."""
    
    def __init__(
        self,
        model_path: str = "yolov8n-face.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda"
    ):
        """
        Initialize face detector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on (cuda/cpu/mps)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self) -> YOLO:
        """Load YOLO model."""
        try:
            # Try to load custom face model
            if self.model_path.exists():
                model = YOLO(str(self.model_path))
            else:
                # Fall back to standard YOLOv8n and use it for face detection
                print(f"Model {self.model_path} not found. Using YOLOv8n...")
                model = YOLO("yolov8n.pt")
            
            # Set device
            import torch
            if self.device == 'cuda' and not torch.cuda.is_available():
                print("CUDA requested but not available. Falling back to CPU.")
                self.device = 'cpu'
            
            model.to(self.device)
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def detect(
        self,
        image: np.ndarray,
        return_image: bool = False
    ) -> Tuple[List[FaceDetection], Optional[np.ndarray]]:
        """
        Detect faces in image.
        
        Args:
            image: Input image (BGR format)
            return_image: Whether to return annotated image
            
        Returns:
            Tuple of (detections, annotated_image)
        """
        # Run inference
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
            classes=[0]  # Only detect person class (or face if using face model)
        )[0]
        
        # Parse detections
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                detection = FaceDetection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=confidence,
                    class_id=class_id
                )
                detections.append(detection)
        
        # Optionally return annotated image
        annotated_image = None
        if return_image:
            annotated_image = results.plot()
        
        return detections, annotated_image
    
    def detect_batch(
        self,
        images: List[np.ndarray]
    ) -> List[List[FaceDetection]]:
        """
        Detect faces in batch of images.
        
        Args:
            images: List of input images (BGR format)
            
        Returns:
            List of detection lists for each image
        """
        # Run batch inference
        results = self.model.predict(
            images,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
            classes=[0]
        )
        
        # Parse all results
        all_detections = []
        for result in results:
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    detection = FaceDetection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=confidence,
                        class_id=class_id
                    )
                    detections.append(detection)
            
            all_detections.append(detections)
        
        return all_detections
    
    def update_parameters(
        self,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ):
        """
        Update detection parameters.
        
        Args:
            confidence_threshold: New confidence threshold
            iou_threshold: New IoU threshold
        """
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
    
    def __repr__(self) -> str:
        return (
            f"FaceDetector(model='{self.model_path}', "
            f"conf={self.confidence_threshold}, "
            f"iou={self.iou_threshold}, "
            f"device='{self.device}')"
        )
