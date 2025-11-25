"""Standalone webcam demo."""

import cv2
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline.vision_pipeline import VisionPipeline
from src.utils.config_loader import get_config


def main():
    """Run webcam demo."""
    # Load config
    config = get_config()
    
    # Initialize pipeline
    print("Initializing vision pipeline...")
    pipeline = VisionPipeline(config)
    
    # Open webcam
    camera_id = config.get('camera', 'default_id', default=0)
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Failed to open camera {camera_id}")
        return
    
    print(f"Started camera {camera_id}")
    print("Controls:")
    print("  q: Quit")
    print("  d: Toggle distance display")
    print("  l: Toggle landmarks display")
    print("  t: Toggle tracking IDs")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = pipeline.process_frame(frame, visualize=True)
            
            # Display
            cv2.imshow('Face Vision System', result.frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                current = config.get('visualization', 'show_distance')
                pipeline.update_config({'visualization': {'show_distance': not current}})
            elif key == ord('l'):
                current = config.get('visualization', 'show_landmarks')
                pipeline.update_config({'visualization': {'show_landmarks': not current}})
            elif key == ord('t'):
                current = config.get('visualization', 'show_track_id')
                pipeline.update_config({'visualization': {'show_track_id': not current}})
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
