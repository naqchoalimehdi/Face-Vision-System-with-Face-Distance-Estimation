"""Test vision pipeline."""

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline.vision_pipeline import VisionPipeline
from src.utils.config_loader import get_config


class TestVisionPipeline(unittest.TestCase):
    """Test VisionPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        # Use CPU for testing to ensure compatibility
        self.config.update({'detection': {'device': 'cpu'}})
        self.pipeline = VisionPipeline(self.config)
        
        # Create a dummy image (black image)
        self.dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a fake face (white rectangle) to test detection if possible
        # Note: YOLO might not detect a simple rectangle, but we test the pipeline flow
        cv2.rectangle(self.dummy_image, (200, 150), (400, 350), (255, 255, 255), -1)
    
    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.detector)
        self.assertIsNotNone(self.pipeline.tracker)
        self.assertIsNotNone(self.pipeline.distance_estimator)
    
    def test_process_frame(self):
        """Test frame processing."""
        result = self.pipeline.process_frame(self.dummy_image, visualize=True)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.frame.shape, self.dummy_image.shape)
        self.assertIsInstance(result.detections, list)
        self.assertIsInstance(result.tracks, list)
        self.assertIsInstance(result.landmarks, list)
        self.assertIsInstance(result.distances, list)
    
    def test_config_update(self):
        """Test configuration update."""
        self.pipeline.update_config({'detection': {'confidence_threshold': 0.8}})
        self.assertEqual(self.pipeline.detector.confidence_threshold, 0.8)


if __name__ == '__main__':
    unittest.main()
