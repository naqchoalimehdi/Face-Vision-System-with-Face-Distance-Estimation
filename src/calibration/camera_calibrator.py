"""Camera calibration utilities."""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict


@dataclass
class CameraCalibration:
    """Camera calibration parameters."""
    camera_matrix: np.ndarray  # 3x3 intrinsic matrix
    dist_coeffs: np.ndarray  # Distortion coefficients
    image_size: Tuple[int, int]  # (width, height)
    reprojection_error: float  # RMS reprojection error
    camera_id: str = "default"


class CameraCalibrator:
    """Camera calibration using checkerboard pattern."""
    
    def __init__(
        self,
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 1.0
    ):
        """
        Initialize camera calibrator.
        
        Args:
            checkerboard_size: Number of inner corners (cols, rows)
            square_size: Size of checkerboard square (in any unit)
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Prepare object points
        self.objp = np.zeros(
            (checkerboard_size[0] * checkerboard_size[1], 3),
            np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0:checkerboard_size[0],
            0:checkerboard_size[1]
        ].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Storage for calibration
        self.obj_points = []  # 3D points in real world
        self.img_points = []  # 2D points in image plane
        self.image_size = None
    
    def add_calibration_image(
        self,
        image: np.ndarray,
        show_corners: bool = False
    ) -> bool:
        """
        Add calibration image.
        
        Args:
            image: Calibration image
            show_corners: Whether to display detected corners
            
        Returns:
            True if checkerboard was found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Store image size
        if self.image_size is None:
            self.image_size = (gray.shape[1], gray.shape[0])
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.checkerboard_size,
            None
        )
        
        if ret:
            # Refine corner positions
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )
            
            # Store points
            self.obj_points.append(self.objp)
            self.img_points.append(corners_refined)
            
            # Optionally show corners
            if show_corners:
                img_with_corners = image.copy()
                cv2.drawChessboardCorners(
                    img_with_corners,
                    self.checkerboard_size,
                    corners_refined,
                    ret
                )
                cv2.imshow('Calibration', img_with_corners)
                cv2.waitKey(500)
            
            return True
        
        return False
    
    def calibrate(self) -> Optional[CameraCalibration]:
        """
        Perform camera calibration.
        
        Returns:
            CameraCalibration object or None if calibration failed
        """
        if len(self.obj_points) < 3:
            print(f"Need at least 3 calibration images, got {len(self.obj_points)}")
            return None
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points,
            self.img_points,
            self.image_size,
            None,
            None
        )
        
        if not ret:
            print("Calibration failed")
            return None
        
        # Compute reprojection error
        total_error = 0
        for i in range(len(self.obj_points)):
            img_points_reprojected, _ = cv2.projectPoints(
                self.obj_points[i],
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs
            )
            error = cv2.norm(
                self.img_points[i],
                img_points_reprojected,
                cv2.NORM_L2
            ) / len(img_points_reprojected)
            total_error += error
        
        mean_error = total_error / len(self.obj_points)
        
        return CameraCalibration(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_size=self.image_size,
            reprojection_error=mean_error
        )
    
    def reset(self):
        """Reset calibration data."""
        self.obj_points.clear()
        self.img_points.clear()
        self.image_size = None
    
    @staticmethod
    def save_calibration(
        calibration: CameraCalibration,
        filepath: str
    ):
        """
        Save calibration to YAML file.
        
        Args:
            calibration: CameraCalibration object
            filepath: Output file path
        """
        data = {
            'camera_id': calibration.camera_id,
            'image_width': calibration.image_size[0],
            'image_height': calibration.image_size[1],
            'camera_matrix': calibration.camera_matrix.tolist(),
            'distortion_coefficients': calibration.dist_coeffs.tolist(),
            'reprojection_error': float(calibration.reprojection_error)
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @staticmethod
    def load_calibration(filepath: str) -> Optional[CameraCalibration]:
        """
        Load calibration from YAML file.
        
        Args:
            filepath: Input file path
            
        Returns:
            CameraCalibration object or None if loading failed
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"Calibration file not found: {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return CameraCalibration(
            camera_matrix=np.array(data['camera_matrix']),
            dist_coeffs=np.array(data['distortion_coefficients']),
            image_size=(data['image_width'], data['image_height']),
            reprojection_error=data['reprojection_error'],
            camera_id=data.get('camera_id', 'default')
        )
    
    def __repr__(self) -> str:
        return (
            f"CameraCalibrator(checkerboard={self.checkerboard_size}, "
            f"images={len(self.obj_points)})"
        )
