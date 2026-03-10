from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TrackState:
    id: int
    center: Tuple[int, int]
    radius: int
    missed: int = 0


class BallTracker:
    def __init__(self, max_distance_px: int = 80, max_missed_frames: int = 20) -> None:
        self.max_distance_px = max_distance_px
        self.max_missed_frames = max_missed_frames
        self.track: Optional[TrackState] = None
        self.next_id = 1
        self.history: List[Tuple[int, int]] = []

    def update(self, detections: List[Tuple[int, int, int]]) -> Optional[TrackState]:
        if not detections:
            if self.track is not None:
                self.track.missed += 1
                if self.track.missed > self.max_missed_frames:
                    self.track = None
            return self.track

        if self.track is None:
            x, y, r = detections[0]
            self.track = TrackState(id=self.next_id, center=(x, y), radius=r)
            self.next_id += 1
            self.history.append((x, y))
            return self.track

        tx, ty = self.track.center
        best_idx = None
        best_dist = float("inf")

        for i, (x, y, _) in enumerate(detections):
            d = np.hypot(x - tx, y - ty)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None and best_dist <= self.max_distance_px:
            x, y, r = detections[best_idx]
            self.track.center = (x, y)
            self.track.radius = r
            self.track.missed = 0
            self.history.append((x, y))
            if len(self.history) > 240:
                self.history = self.history[-240:]
        else:
            self.track.missed += 1
            if self.track.missed > self.max_missed_frames:
                self.track = None

        return self.track
