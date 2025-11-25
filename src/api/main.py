"""FastAPI application entry point."""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .routes import detection, camera, calibration
from .websocket import websocket_endpoint
from ..utils.config_loader import get_config


# Create FastAPI app
app = FastAPI(
    title="Face Vision System API",
    description="Real-time face detection, tracking, and distance estimation",
    version="1.0.0"
)

# Configure CORS
config = get_config()
api_config = config.get_section('api')

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.get('cors_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(detection.router)
app.include_router(camera.router)
app.include_router(calibration.router)

# WebSocket endpoint
@app.websocket("/ws/stream")
async def stream(websocket: WebSocket, camera_id: int = 0):
    """WebSocket endpoint for video streaming."""
    await websocket_endpoint(websocket, camera_id)

# Serve static files (web UI)
web_dir = Path(__file__).parent.parent.parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    print("🚀 Face Vision System API starting...")
    print(f"📷 API documentation: http://localhost:{api_config.get('port', 8000)}/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    print("👋 Face Vision System API shutting down...")
    
    # Clean up resources
    from .routes.camera import active_cameras
    for cap in active_cameras.values():
        cap.release()
