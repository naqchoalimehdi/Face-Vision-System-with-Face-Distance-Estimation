"""Multi-object tracker for managing multiple face tracks."""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from .kalman_tracker import KalmanTracker


@dataclass
class Track:
    """Face track with metadata."""
    track_id: int
    kalman: KalmanTracker
    bbox: Tuple[int, int, int, int]
    confidence: float
    age: int
    hits: int
    time_since_update: int


class MultiTracker:
    """Multi-object tracker for faces."""
    
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 50.0,
        min_hits: int = 3
    ):
        """
        Initialize multi-tracker.
        
        Args:
            max_disappeared: Max frames before removing track
            max_distance: Max distance for track association (pixels)
            min_hits: Minimum hits before track is confirmed
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits
        
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0
    
    def update(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], float]]
    ) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of (bbox, confidence) tuples
            
        Returns:
            List of active tracks
        """
        # Predict all existing tracks
        for track in self.tracks.values():
            track.kalman.predict()
            track.time_since_update += 1
        
        # Associate detections with tracks
        if len(detections) > 0 and len(self.tracks) > 0:
            matched, unmatched_dets, unmatched_tracks = self._associate(detections)
            
            # Update matched tracks
            for det_idx, track_id in matched:
                bbox, confidence = detections[det_idx]
                track = self.tracks[track_id]
                track.kalman.update(bbox)
                track.bbox = track.kalman.get_bbox()
                track.confidence = confidence
                track.hits += 1
                track.time_since_update = 0
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                bbox, confidence = detections[det_idx]
                self._create_track(bbox, confidence)
            
            # Mark unmatched tracks
            for track_id in unmatched_tracks:
                # Track already updated in predict step
                pass
        
        elif len(detections) > 0:
            # No existing tracks, create new ones
            for bbox, confidence in detections:
                self._create_track(bbox, confidence)
        
        # Remove old tracks
        self._remove_old_tracks()
        
        # Return confirmed tracks
        confirmed_tracks = [
            track for track in self.tracks.values()
            if track.hits >= self.min_hits
        ]
        
        return confirmed_tracks
    
    def _associate(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], float]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with tracks using IoU or distance.
        
        Args:
            detections: List of (bbox, confidence) tuples
            
        Returns:
            Tuple of (matched, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Compute cost matrix (using centroid distance)
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(detections), len(track_ids)))
        
        for i, (det_bbox, _) in enumerate(detections):
            det_center = self._get_bbox_center(det_bbox)
            
            for j, track_id in enumerate(track_ids):
                track_center = self.tracks[track_id].kalman.get_center()
                distance = np.linalg.norm(
                    np.array(det_center) - np.array(track_center)
                )
                cost_matrix[i, j] = distance
        
        # Simple greedy matching
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = track_ids.copy()
        
        # Find best matches
        while len(unmatched_dets) > 0 and len(unmatched_tracks) > 0:
            # Find minimum cost
            min_cost = float('inf')
            min_det_idx = -1
            min_track_idx = -1
            
            for i in unmatched_dets:
                for j, track_id in enumerate(unmatched_tracks):
                    track_idx = track_ids.index(track_id)
                    if cost_matrix[i, track_idx] < min_cost:
                        min_cost = cost_matrix[i, track_idx]
                        min_det_idx = i
                        min_track_idx = j
            
            # Check if cost is acceptable
            if min_cost < self.max_distance:
                track_id = unmatched_tracks[min_track_idx]
                matched.append((min_det_idx, track_id))
                unmatched_dets.remove(min_det_idx)
                unmatched_tracks.remove(track_id)
            else:
                break
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _create_track(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float
    ):
        """Create new track."""
        kalman = KalmanTracker(bbox)
        
        track = Track(
            track_id=self.next_track_id,
            kalman=kalman,
            bbox=bbox,
            confidence=confidence,
            age=0,
            hits=1,
            time_since_update=0
        )
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
    
    def _remove_old_tracks(self):
        """Remove tracks that haven't been updated."""
        to_remove = []
        
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_disappeared:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def _get_bbox_center(
        self,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[float, float]:
        """Get bounding box center."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return (cx, cy)
    
    def _compute_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id)
    
    def reset(self):
        """Reset all tracks."""
        self.tracks.clear()
        self.next_track_id = 0
    
    def __len__(self) -> int:
        return len(self.tracks)
    
    def __repr__(self) -> str:
        return f"MultiTracker(tracks={len(self.tracks)}, next_id={self.next_track_id})"
