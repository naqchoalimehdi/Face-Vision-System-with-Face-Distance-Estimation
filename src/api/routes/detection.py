"""Detection API routes."""

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import io
from PIL import Image

from ..models.schemas import DetectionResponse, Detection, BoundingBox, ErrorResponse
from ...pipeline.vision_pipeline import VisionPipeline
from ...utils.config_loader import get_config


router = APIRouter(prefix="/api/detect", tags=["detection"])

# Global pipeline instance
pipeline = None


def get_pipeline() -> VisionPipeline:
    """Get or create pipeline instance."""
    global pipeline
    if pipeline is None:
        config = get_config()
        pipeline = VisionPipeline(config)
    return pipeline


@router.post("/image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    """
    Detect faces in uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Detection results
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image
        pipe = get_pipeline()
        result = pipe.process_frame(image, visualize=False)
        
        # Format response
        detections = []
        for i, track in enumerate(result.tracks):
            x1, y1, x2, y2 = track.bbox
            
            detection = Detection(
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=track.confidence,
                track_id=track.track_id,
                distance_cm=result.distances[i] if i < len(result.distances) else None
            )
            detections.append(detection)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            frame_count=result.frame_count,
            fps=result.fps
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Process uploaded video file.
    
    Args:
        file: Uploaded video file
        
    Returns:
        Job ID for tracking processing status
    """
    # TODO: Implement async video processing with job queue
    raise HTTPException(status_code=501, detail="Video processing not yet implemented")


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Get processing status for a job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Job status
    """
    # TODO: Implement job status tracking
    raise HTTPException(status_code=501, detail="Job tracking not yet implemented")
