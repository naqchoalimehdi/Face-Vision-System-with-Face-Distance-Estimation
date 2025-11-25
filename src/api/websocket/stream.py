"""WebSocket streaming handler for real-time video."""

import cv2
import numpy as np
import asyncio
import base64
from fastapi import WebSocket, WebSocketDisconnect
import json
from typing import Optional

from ...pipeline.vision_pipeline import VisionPipeline
from ...utils.config_loader import get_config
from ..routes.camera import get_active_camera, get_camera_lock


class StreamManager:
    """Manage WebSocket video streaming."""
    
    def __init__(self):
        self.pipeline: Optional[VisionPipeline] = None
        self.active_connections: list[WebSocket] = []
    
    def get_pipeline(self) -> VisionPipeline:
        """Get or create pipeline instance."""
        if self.pipeline is None:
            config = get_config()
            self.pipeline = VisionPipeline(config)
        return self.pipeline
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def stream_camera(
        self,
        websocket: WebSocket,
        camera_id: int,
        fps: int = 30
    ):
        """
        Stream camera feed through WebSocket.
        
        Args:
            websocket: WebSocket connection
            camera_id: Camera index
            fps: Target FPS
        """
        try:
            # Get camera
            cap = get_active_camera(camera_id)
            lock = get_camera_lock(camera_id)
            
            # Get pipeline
            pipeline = self.get_pipeline()
            
            # Calculate frame interval
            frame_interval = 1.0 / fps
            
            while True:
                start_time = asyncio.get_event_loop().time()
                
                # Read frame
                with lock:
                    ret, frame = cap.read()
                
                if not ret:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to read frame"
                    })
                    break
                
                # Process frame
                result = pipeline.process_frame(frame, visualize=True)
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', result.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Prepare detection data
                detections = []
                for i, track in enumerate(result.tracks):
                    x1, y1, x2, y2 = track.bbox
                    detections.append({
                        "track_id": track.track_id,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "confidence": track.confidence,
                        "distance_cm": result.distances[i] if i < len(result.distances) else None
                    })
                
                # Send frame and data
                await websocket.send_json({
                    "type": "frame",
                    "frame": frame_base64,
                    "detections": detections,
                    "fps": result.fps,
                    "frame_count": result.frame_count
                })
                
                # Wait for next frame
                elapsed = asyncio.get_event_loop().time() - start_time
                wait_time = max(0, frame_interval - elapsed)
                await asyncio.sleep(wait_time)
        
        except WebSocketDisconnect:
            self.disconnect(websocket)
        except Exception as e:
            print(f"Streaming error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except:
                pass
            self.disconnect(websocket)


# Global stream manager
stream_manager = StreamManager()


async def websocket_endpoint(websocket: WebSocket, camera_id: int = 0):
    """
    WebSocket endpoint for video streaming.
    
    Args:
        websocket: WebSocket connection
        camera_id: Camera index
    """
    await stream_manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "message": f"Connected to camera {camera_id}"
        })
        
        # Start streaming
        await stream_manager.stream_camera(websocket, camera_id)
    
    except WebSocketDisconnect:
        stream_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        stream_manager.disconnect(websocket)
