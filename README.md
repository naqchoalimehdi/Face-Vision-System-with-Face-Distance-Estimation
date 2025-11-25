# Face Vision System

A high-performance, real-time face detection, tracking, and distance estimation system powered by YOLOv8 and MediaPipe.

![Face Vision System](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png)

## Features

- **Real-time Face Detection**: Uses YOLOv8 for fast and accurate face detection.
- **Smooth Tracking**: Implements Kalman filtering for stable face tracking across frames.
- **Facial Landmarks**: Detects 468 facial landmarks using MediaPipe.
- **Distance Estimation**: Estimates distance to faces using facial width or 3D pose estimation.
- **Web Interface**: Modern, dark-mode Web UI with real-time video streaming.
- **API**: FastAPI-based REST API and WebSocket streaming.
- **Multi-Camera Support**: Easily switch between available cameras.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd face-vision-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the Web Server (Recommended)

Start the full system with Web UI and API:

```bash
python examples/run_server.py
```

- **Web UI**: Open [http://localhost:8000](http://localhost:8000) in your browser.
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Standalone Webcam Demo

Run the vision pipeline locally without the web server:

```bash
python examples/webcam_demo.py
```

**Controls:**
- `q`: Quit
- `d`: Toggle distance display
- `l`: Toggle landmarks display
- `t`: Toggle tracking IDs

## Configuration

The system is highly configurable via `config/default_config.yaml`. You can adjust:

- **Detection**: Model path, confidence threshold, IoU threshold.
- **Tracking**: Kalman filter parameters, max disappeared frames.
- **Distance**: Estimation method (facial width vs pose), average face width.
- **Camera**: Default ID, resolution, FPS.

## Project Structure

```
face-vision-system/
├── config/                 # Configuration files
├── examples/               # Example scripts
├── src/
│   ├── api/                # FastAPI backend
│   ├── calibration/        # Camera calibration
│   ├── detection/          # YOLO & MediaPipe detectors
│   ├── distance/           # Distance estimation
│   ├── pipeline/           # Main vision pipeline
│   ├── tracking/           # Kalman tracker
│   └── utils/              # Utilities
├── web/                    # Web UI (HTML/CSS/JS)
├── requirements.txt
└── README.md
```

## Technologies

- **YOLOv8** (Ultralytics)
- **MediaPipe** (Google)
- **OpenCV**
- **FastAPI**
- **NumPy & SciPy**

## License

MIT License
