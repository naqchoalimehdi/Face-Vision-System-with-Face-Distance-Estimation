"""Calibration API routes."""

from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List
import os

from ..models.schemas import CalibrationStatus, CalibrationProfile
from ...calibration.camera_calibrator import CameraCalibrator


router = APIRouter(prefix="/api/calibration", tags=["calibration"])

# Global calibrator instance
calibrator: CameraCalibrator = None
calibration_active = False


@router.post("/start")
async def start_calibration():
    """
    Start calibration process.
    
    Returns:
        Success message
    """
    global calibrator, calibration_active
    
    calibrator = CameraCalibrator()
    calibration_active = True
    
    return {"success": True, "message": "Calibration started"}


@router.post("/capture")
async def capture_calibration_frame():
    """
    Capture calibration frame.
    
    Returns:
        Capture status
    """
    global calibrator, calibration_active
    
    if not calibration_active or calibrator is None:
        raise HTTPException(status_code=400, detail="Calibration not started")
    
    # TODO: Implement frame capture from active camera
    raise HTTPException(status_code=501, detail="Frame capture not yet implemented")


@router.post("/compute")
async def compute_calibration(camera_id: str = "default"):
    """
    Compute calibration parameters.
    
    Args:
        camera_id: Camera identifier
        
    Returns:
        Calibration results
    """
    global calibrator, calibration_active
    
    if not calibration_active or calibrator is None:
        raise HTTPException(status_code=400, detail="Calibration not started")
    
    # Perform calibration
    calibration = calibrator.calibrate()
    
    if calibration is None:
        raise HTTPException(status_code=400, detail="Calibration failed")
    
    # Save calibration
    calibration.camera_id = camera_id
    output_path = Path("config") / f"camera_calibration_{camera_id}.yaml"
    CameraCalibrator.save_calibration(calibration, str(output_path))
    
    calibration_active = False
    
    return {
        "success": True,
        "reprojection_error": calibration.reprojection_error,
        "profile_path": str(output_path)
    }


@router.get("/profiles", response_model=List[CalibrationProfile])
async def list_calibration_profiles():
    """
    List saved calibration profiles.
    
    Returns:
        List of calibration profiles
    """
    profiles = []
    config_dir = Path("config")
    
    if config_dir.exists():
        for file in config_dir.glob("camera_calibration_*.yaml"):
            try:
                from ...calibration.camera_calibrator import CameraCalibrator
                calib = CameraCalibrator.load_calibration(str(file))
                
                if calib:
                    profiles.append(CalibrationProfile(
                        camera_id=calib.camera_id,
                        image_width=calib.image_size[0],
                        image_height=calib.image_size[1],
                        reprojection_error=calib.reprojection_error,
                        created_at=None  # Could add file modification time
                    ))
            except Exception as e:
                print(f"Failed to load calibration {file}: {e}")
    
    return profiles


@router.get("/status", response_model=CalibrationStatus)
async def get_calibration_status():
    """
    Get current calibration status.
    
    Returns:
        Calibration status
    """
    global calibrator, calibration_active
    
    if not calibration_active or calibrator is None:
        return CalibrationStatus(
            is_calibrating=False,
            images_captured=0,
            images_required=10
        )
    
    return CalibrationStatus(
        is_calibrating=True,
        images_captured=len(calibrator.obj_points),
        images_required=10
    )
