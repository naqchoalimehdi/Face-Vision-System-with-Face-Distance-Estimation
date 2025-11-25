"""Kalman filter for smooth face tracking."""

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import block_diag


class KalmanTracker:
    """Kalman filter for tracking face position and velocity."""
    
    def __init__(
        self,
        initial_bbox: Tuple[int, int, int, int],
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        dt: float = 1.0
    ):
        """
        Initialize Kalman tracker.
        
        Args:
            initial_bbox: Initial bounding box (x1, y1, x2, y2)
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
            dt: Time step
        """
        self.dt = dt
        
        # State: [x_center, y_center, width, height, vx, vy, vw, vh]
        x1, y1, x2, y2 = initial_bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only observe position and size)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Process noise covariance
        q = process_noise
        self.Q = block_diag(
            q * np.eye(4),  # Position and size
            q * 10 * np.eye(4)  # Velocity (higher uncertainty)
        )
        
        # Measurement noise covariance
        r = measurement_noise
        self.R = r * np.eye(4)
        
        # State covariance
        self.P = np.eye(8) * 1000  # High initial uncertainty
        
        # Track metadata
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
    
    def predict(self) -> np.ndarray:
        """
        Predict next state.
        
        Returns:
            Predicted state
        """
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.age += 1
        self.time_since_update += 1
        
        return self.state
    
    def update(self, measurement: Tuple[int, int, int, int]):
        """
        Update state with measurement.
        
        Args:
            measurement: Measured bounding box (x1, y1, x2, y2)
        """
        # Convert bbox to center-size format
        x1, y1, x2, y2 = measurement
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        z = np.array([cx, cy, w, h])
        
        # Innovation
        y = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P
        
        # Update metadata
        self.hits += 1
        self.time_since_update = 0
    
    def get_bbox(self) -> Tuple[int, int, int, int]:
        """
        Get current bounding box.
        
        Returns:
            Bounding box (x1, y1, x2, y2)
        """
        cx, cy, w, h = self.state[:4]
        
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        return (x1, y1, x2, y2)
    
    def get_center(self) -> Tuple[float, float]:
        """
        Get current center position.
        
        Returns:
            Center coordinates (cx, cy)
        """
        return (self.state[0], self.state[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity.
        
        Returns:
            Velocity (vx, vy)
        """
        return (self.state[4], self.state[5])
    
    def __repr__(self) -> str:
        bbox = self.get_bbox()
        return f"KalmanTracker(bbox={bbox}, age={self.age}, hits={self.hits})"
