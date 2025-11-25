"""Camera management API routes."""

import cv2
from fastapi import APIRouter, HTTPException
from typing import List, Dict
import threading

from ..models.schemas import CameraInfo, CameraConfig, ErrorResponse


router = APIRouter(prefix="/api/cameras", tags=["camera"])

# Active camera streams
active_cameras: Dict[int, cv2.VideoCapture] = {}
camera_locks: Dict[int, threading.Lock] = {}


@router.get("", response_model=List[CameraInfo])
async def list_cameras():
    """
    List available cameras.
    
    Returns:
        List of available cameras
    """
    cameras = []
    
    # Try to detect cameras (check first 5 indices)
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            cameras.append(CameraInfo(
                camera_id=i,
                name=f"Camera {i}",
                width=width,
                height=height,
                fps=fps if fps > 0 else 30.0,
                is_active=i in active_cameras
            ))
            
            cap.release()
    
    return cameras


@router.post("/{camera_id}/start")
async def start_camera(camera_id: int):
    """
    Start camera stream.
    
    Args:
        camera_id: Camera index
        
    Returns:
        Success message
    """
    if camera_id in active_cameras:
        return {"success": True, "message": "Camera already active"}
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    active_cameras[camera_id] = cap
    camera_locks[camera_id] = threading.Lock()
    
    return {"success": True, "message": f"Camera {camera_id} started"}


@router.post("/{camera_id}/stop")
async def stop_camera(camera_id: int):
    """
    Stop camera stream.
    
    Args:
        camera_id: Camera index
        
    Returns:
        Success message
    """
    if camera_id not in active_cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not active")
    
    cap = active_cameras[camera_id]
    cap.release()
    
    del active_cameras[camera_id]
    del camera_locks[camera_id]
    
    return {"success": True, "message": f"Camera {camera_id} stopped"}


@router.get("/{camera_id}/config")
async def get_camera_config(camera_id: int):
    """
    Get camera configuration.
    
    Args:
        camera_id: Camera index
        
    Returns:
        Camera configuration
    """
    if camera_id not in active_cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not active")
    
    cap = active_cameras[camera_id]
    
    return CameraConfig(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=int(cap.get(cv2.CAP_PROP_FPS))
    )


@router.put("/{camera_id}/config")
async def update_camera_config(camera_id: int, config: CameraConfig):
    """
    Update camera configuration.
    
    Args:
        camera_id: Camera index
        config: New configuration
        
    Returns:
        Success message
    """
    if camera_id not in active_cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not active")
    
    cap = active_cameras[camera_id]
    
    if config.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    
    if config.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    
    if config.fps is not None:
        cap.set(cv2.CAP_PROP_FPS, config.fps)
    
    return {"success": True, "message": "Camera configuration updated"}


def get_active_camera(camera_id: int) -> cv2.VideoCapture:
    """Get active camera capture object."""
    if camera_id not in active_cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not active")
    return active_cameras[camera_id]


def get_camera_lock(camera_id: int) -> threading.Lock:
    """Get camera lock."""
    if camera_id not in camera_locks:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not active")
    return camera_locks[camera_id]
