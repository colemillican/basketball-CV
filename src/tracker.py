from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TrackState:
    id: int
    center: Tuple[int, int]
    radius: int
    missed: int = 0


class BallTracker:
    # Stop extrapolating after this many consecutive missed frames.
    # Beyond this the velocity estimate is too stale to be useful.
    MAX_EXTRAPOLATE_FRAMES: int = 8

    def __init__(self, max_distance_px: int = 80, max_missed_frames: int = 20) -> None:
        self.max_distance_px = max_distance_px
        self.max_missed_frames = max_missed_frames
        self.track: Optional[TrackState] = None
        self.next_id = 1
        self.history: List[Tuple[int, int]] = []

        # Dead reckoning state
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        # Unique positions only (filters repeated detections from threaded YOLO)
        self._unique_history: List[Tuple[int, int]] = []

    def update(self, detections: List[Tuple[int, int, int]]) -> Optional[TrackState]:
        if not detections:
            if self.track is not None:
                self.track.missed += 1
                if self.track.missed > self.max_missed_frames:
                    self.track = None
                elif self.track.missed <= self.MAX_EXTRAPOLATE_FRAMES:
                    # Dead reckoning: advance position by current velocity
                    cx, cy = self.track.center
                    vx, vy = self._velocity
                    self.track.center = (int(cx + vx), int(cy + vy))
                    # Dampen velocity slightly to model deceleration
                    self._velocity = (vx * 0.9, vy * 0.9)
            return self.track

        if self.track is None:
            x, y, r = detections[0]
            self.track = TrackState(id=self.next_id, center=(x, y), radius=r)
            self.next_id += 1
            self.history.append((x, y))
            self._unique_history.append((x, y))
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

            # Update velocity only when position meaningfully changed
            # (filters out repeated detections from threaded YOLO)
            if self._unique_history:
                lx, ly = self._unique_history[-1]
                if np.hypot(x - lx, y - ly) > 2.0:
                    new_vx, new_vy = float(x - lx), float(y - ly)
                    old_vx, old_vy = self._velocity
                    self._velocity = (
                        0.7 * new_vx + 0.3 * old_vx,
                        0.7 * new_vy + 0.3 * old_vy,
                    )
                    self._unique_history.append((x, y))
                    if len(self._unique_history) > 30:
                        self._unique_history = self._unique_history[-30:]
            else:
                self._unique_history.append((x, y))
        else:
            self.track.missed += 1
            if self.track.missed > self.max_missed_frames:
                self.track = None
            elif self.track.missed <= self.MAX_EXTRAPOLATE_FRAMES:
                cx, cy = self.track.center
                vx, vy = self._velocity
                self.track.center = (int(cx + vx), int(cy + vy))
                self._velocity = (vx * 0.9, vy * 0.9)

        return self.track
