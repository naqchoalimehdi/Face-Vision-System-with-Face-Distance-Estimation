from ultralytics import YOLO
import sys

def verify_model():
    print("Attempting to load YOLOv8n model...")
    try:
        model = YOLO("yolov8n.pt")
        print("Successfully loaded yolov8n.pt")
    except Exception as e:
        print(f"Failed to load yolov8n.pt: {e}")
        sys.exit(1)

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    verify_model()
