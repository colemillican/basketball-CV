from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ShotAttempt:
    id: int
    start_frame: int
    start_px: Tuple[int, int]
    result: str = "pending"
    end_frame: int = -1
    confidence: float = 0.0
    signal_breakdown: Dict[str, float] = field(default_factory=dict)


class ShotLogic:
    """
    Detects basketball shot attempts and classifies them as make or miss.

    Two scoring modes (controlled by use_confidence_scoring):

    Legacy mode  — original AND logic: all three conditions must be true.
                   (seen_above_rim AND inside_rim N frames AND net_lane N frames)

    Confidence mode — weighted multi-signal model. Each signal contributes a
                   score; score >= make_threshold → make. Signals:
                     +20  above_rim_entry   ball seen above rim plane
                     +20  inside_rim_zone   ball held inside scoring zone
                     +35  net_lane_passage  ball descended through net lane
                     +20  net_optical_flow  motion in net ROI (optical flow)
                     -50  bounce_away       ball moved away from rim after entry
                   Max positive score ≈ 95; default threshold = 60.

    Camera assumption: fixed mount at top of backboard, angled slightly downward.
    """

    def __init__(
        self,
        rim_center: Tuple[int, int],
        rim_radius_px: int,
        score_cooldown_frames: int = 15,
        make_confirm_frames: int = 3,
        miss_timeout_frames: int = 55,
        min_shot_arc_drop_px: int = 28,
        entry_above_margin_px: int = 8,
        net_drop_margin_px: int = 14,
        net_lane_radius_scale: float = 0.8,
        net_confirm_frames: int = 2,
        # Confidence scoring params
        use_confidence_scoring: bool = True,
        make_threshold: float = 60.0,
        net_flow_threshold: float = 1.5,
        min_tracking_frames: int = 10,
        min_travel_px: float = 150.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.rim_center = rim_center
        self.rim_radius_px = rim_radius_px
        self.score_cooldown_frames = score_cooldown_frames
        self.make_confirm_frames = make_confirm_frames
        self.miss_timeout_frames = miss_timeout_frames
        self.min_shot_arc_drop_px = min_shot_arc_drop_px
        self.entry_above_margin_px = entry_above_margin_px
        self.net_drop_margin_px = net_drop_margin_px
        self.net_lane_radius_scale = net_lane_radius_scale
        self.net_confirm_frames = net_confirm_frames

        self.use_confidence_scoring = use_confidence_scoring
        self.make_threshold = make_threshold
        self.net_flow_threshold = net_flow_threshold
        self.min_tracking_frames = min_tracking_frames
        self.min_travel_px = min_travel_px

        _default_weights: Dict[str, float] = {
            "above_rim_entry":  20.0,
            "inside_rim_zone":  20.0,
            "net_lane_passage": 35.0,
            "net_optical_flow": 20.0,
            "bounce_away":     -50.0,
        }
        self.weights = {**_default_weights, **(weights or {})}

        # Per-attempt tracking state
        self.frame_idx = 0
        self.attempt_id = 0
        self.current: Optional[ShotAttempt] = None
        self.recent_positions: List[Tuple[int, int]] = []
        self.inside_rim_frames = 0
        self.net_lane_frames = 0
        self.seen_above_rim = False
        self.last_ball_center: Optional[Tuple[int, int]] = None
        self.last_scored_frame = -10_000

        # Confidence scoring state (reset per attempt)
        self._signals: Dict[str, float] = {}
        self._max_net_flow: float = 0.0
        self._tracking_frames_since_launch: int = 0
        self._travel_px_since_launch: float = 0.0
        self._consecutive_outside: int = 0

        # Frame-level state (persists across attempts for optical flow)
        self._prev_gray: Optional[np.ndarray] = None

        # Net ROI bounding box derived from rim geometry
        rx, ry = rim_center
        lane_half = int(rim_radius_px * net_lane_radius_scale)
        self._net_roi: Tuple[int, int, int, int] = (
            rx - lane_half,
            ry + net_drop_margin_px,
            rx + lane_half,
            ry + rim_radius_px * 2,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        ball_center: Optional[Tuple[int, int]],
        frame: Optional[np.ndarray] = None,
    ) -> Optional[ShotAttempt]:
        """
        Call once per frame.

        Args:
            ball_center: tracked ball pixel position, or None if not detected.
            frame:       current BGR frame (optional). Required for optical flow
                         signal; safe to omit when optical flow is not needed.

        Returns a completed ShotAttempt when a shot resolves, otherwise None.
        """
        self.frame_idx += 1

        if frame is not None and self.current is not None:
            self._update_net_flow(frame)

        outcome = self._step(ball_center)

        if frame is not None:
            self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return outcome

    def debug_state(self) -> Dict:
        rx, ry = self.rim_center
        lane_half = int(self.rim_radius_px * self.net_lane_radius_scale)
        return {
            "rim_center": (rx, ry),
            "scoring_radius": int(self.rim_radius_px * 1.1),
            "entry_line_y": int(ry - self.entry_above_margin_px),
            "net_line_y": int(ry + self.net_drop_margin_px),
            "net_lane_x_left": int(rx - lane_half),
            "net_lane_x_right": int(rx + lane_half),
            "inside_rim_frames": self.inside_rim_frames,
            "net_lane_frames": self.net_lane_frames,
            "seen_above_rim": self.seen_above_rim,
            "has_active_attempt": self.current is not None,
            "attempt_id": None if self.current is None else self.current.id,
            "confidence_score": round(sum(self._signals.values()), 1),
            "signals": dict(self._signals),
            "max_net_flow": round(self._max_net_flow, 3),
        }

    # ------------------------------------------------------------------
    # Core step logic
    # ------------------------------------------------------------------

    def _step(self, ball_center: Optional[Tuple[int, int]]) -> Optional[ShotAttempt]:
        if ball_center is None:
            return self._handle_no_detection()

        self.recent_positions.append(ball_center)
        if len(self.recent_positions) > 30:
            self.recent_positions = self.recent_positions[-30:]

        if self.current is None:
            self._try_start_attempt(ball_center)
            self.last_ball_center = ball_center
            return None

        return self._update_active_attempt(ball_center)

    def _handle_no_detection(self) -> Optional[ShotAttempt]:
        if self.current is None:
            self.last_ball_center = None
            return None

        elapsed = self.frame_idx - self.current.start_frame
        if elapsed > self.miss_timeout_frames:
            return self._close_attempt("miss")

        self.last_ball_center = None
        return None

    def _try_start_attempt(self, ball_center: Tuple[int, int]) -> None:
        if self.frame_idx - self.last_scored_frame < self.score_cooldown_frames:
            return
        if len(self.recent_positions) < 6:
            return

        ys = [p[1] for p in self.recent_positions[-6:]]
        arc_drop = ys[-1] - min(ys)
        if self._is_in_launch_region(ball_center) and arc_drop > self.min_shot_arc_drop_px:
            self.attempt_id += 1
            self.current = ShotAttempt(
                id=self.attempt_id,
                start_frame=self.frame_idx,
                start_px=self.recent_positions[-6],
            )
            self._reset_attempt_state()

    def _update_active_attempt(self, ball_center: Tuple[int, int]) -> Optional[ShotAttempt]:
        # Track per-attempt travel for gate condition
        if self.last_ball_center is not None:
            dx = ball_center[0] - self.last_ball_center[0]
            dy = ball_center[1] - self.last_ball_center[1]
            self._travel_px_since_launch += (dx * dx + dy * dy) ** 0.5
        self._tracking_frames_since_launch += 1

        # Above-rim check
        if self._is_above_rim(ball_center):
            self.seen_above_rim = True

        # Scoring zone (rim proximity)
        if self._is_in_scoring_zone(ball_center):
            self.inside_rim_frames += 1
        else:
            self.inside_rim_frames = 0

        # Net lane: ball must be moving downward through lane
        downward = (
            self.last_ball_center is not None
            and ball_center[1] > self.last_ball_center[1]
        )
        if self._is_in_net_lane(ball_center) and downward:
            self.net_lane_frames += 1
        else:
            self.net_lane_frames = 0

        is_make = self._evaluate_make(ball_center)

        if is_make:
            return self._close_attempt("make")

        if self.frame_idx - self.current.start_frame > self.miss_timeout_frames:
            return self._close_attempt("miss")

        self.last_ball_center = ball_center
        return None

    # ------------------------------------------------------------------
    # Make evaluation
    # ------------------------------------------------------------------

    def _evaluate_make(self, ball_center: Tuple[int, int]) -> bool:
        if self.use_confidence_scoring:
            return self._confidence_is_make(ball_center)
        return self._legacy_is_make()

    def _legacy_is_make(self) -> bool:
        return (
            self.seen_above_rim
            and self.inside_rim_frames >= self.make_confirm_frames
            and self.net_lane_frames >= self.net_confirm_frames
        )

    def _confidence_is_make(self, ball_center: Tuple[int, int]) -> bool:
        # Gate: discard if trajectory too short (filters loose balls)
        if (
            self._tracking_frames_since_launch < self.min_tracking_frames
            or self._travel_px_since_launch < self.min_travel_px
        ):
            return False

        # --- Positive signals ---
        if self.seen_above_rim:
            self._signals["above_rim_entry"] = self.weights["above_rim_entry"]

        if self.inside_rim_frames >= self.make_confirm_frames:
            self._signals["inside_rim_zone"] = self.weights["inside_rim_zone"]

        if self.net_lane_frames >= self.net_confirm_frames:
            self._signals["net_lane_passage"] = self.weights["net_lane_passage"]

        if self._max_net_flow >= self.net_flow_threshold:
            self._signals["net_optical_flow"] = self.weights["net_optical_flow"]

        # --- Negative signal: ball moving away from rim ---
        if (
            self.last_ball_center is not None
            and not self._is_in_scoring_zone(ball_center)
        ):
            prev_dist = self._distance_to_rim(self.last_ball_center)
            curr_dist = self._distance_to_rim(ball_center)
            if curr_dist > prev_dist + 5:
                self._consecutive_outside += 1
                if self._consecutive_outside >= 3:
                    self._signals["bounce_away"] = self.weights["bounce_away"]
            else:
                self._consecutive_outside = 0

        return sum(self._signals.values()) >= self.make_threshold

    # ------------------------------------------------------------------
    # Attempt resolution
    # ------------------------------------------------------------------

    def _close_attempt(self, result: str) -> ShotAttempt:
        assert self.current is not None
        self.current.result = result
        self.current.end_frame = self.frame_idx
        self.current.confidence = round(sum(self._signals.values()), 1)
        self.current.signal_breakdown = dict(self._signals)
        done = self.current
        self.current = None
        self.last_scored_frame = self.frame_idx
        self._reset_attempt_state()
        self.last_ball_center = None
        return done

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _distance_to_rim(self, p: Tuple[int, int]) -> float:
        rx, ry = self.rim_center
        return ((p[0] - rx) ** 2 + (p[1] - ry) ** 2) ** 0.5

    def _is_in_launch_region(self, p: Tuple[int, int]) -> bool:
        return self._distance_to_rim(p) > self.rim_radius_px * 2.2

    def _is_in_scoring_zone(self, p: Tuple[int, int]) -> bool:
        return self._distance_to_rim(p) <= self.rim_radius_px * 1.1

    def _is_above_rim(self, p: Tuple[int, int]) -> bool:
        _, ry = self.rim_center
        return p[1] <= ry - self.entry_above_margin_px

    def _is_in_net_lane(self, p: Tuple[int, int]) -> bool:
        rx, ry = self.rim_center
        lane_half = self.rim_radius_px * self.net_lane_radius_scale
        return abs(p[0] - rx) <= lane_half and p[1] >= ry + self.net_drop_margin_px

    # ------------------------------------------------------------------
    # Optical flow
    # ------------------------------------------------------------------

    def _update_net_flow(self, frame: np.ndarray) -> None:
        if self._prev_gray is None:
            return
        x1, y1, x2, y2 = self._net_roi
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_curr = gray[y1:y2, x1:x2]
        roi_prev = self._prev_gray[y1:y2, x1:x2]
        if roi_curr.size == 0 or roi_prev.size == 0:
            return
        flow = cv2.calcOpticalFlowFarneback(
            roi_prev, roi_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        self._max_net_flow = max(self._max_net_flow, float(np.mean(mag)))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_attempt_state(self) -> None:
        self.inside_rim_frames = 0
        self.net_lane_frames = 0
        self.seen_above_rim = False
        self._signals = {}
        self._max_net_flow = 0.0
        self._tracking_frames_since_launch = 0
        self._travel_px_since_launch = 0.0
        self._consecutive_outside = 0
