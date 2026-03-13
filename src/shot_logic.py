"""
Top-down backboard camera shot logic — v2

CAMERA GEOMETRY
---------------
Camera is mounted flush behind / above the backboard, angled ~15° downward.
The rim appears as a large circle in the lower-center of frame.
The court and players appear in the upper portion of the frame.

    ┌──────────────────────────────────┐
    │  (court / players — upper frame) │  ← ball enters from here
    │                                  │
    │         ○  ○                     │
    │       ○      ○   ← halo zone     │
    │      ○  ●──── ○  ← rim circle    │
    │      ○  (net) ○                  │
    │       ○      ○                   │
    └──────────────────────────────────┘

A MAKE from this angle
  1. Ball appears in upper frame (court side), moving toward rim.
  2. Ball enters the halo zone  (within HALO_SCALE × rim_radius).
  3. Ball enters the scoring zone (inside the rim circle).
  4. Ball DISAPPEARS — it has fallen below the rim plane into the net.
     The net interior shows a pixel-change spike.

A MISS from this angle
  1. Ball approaches rim.
  2. Ball contacts the rim edge.
  3. Ball velocity REVERSES — moves away from rim center after contact.

False-positive prevention
  - Dribbles:     ball oscillates near the player, never sustains a long
                  approach toward the rim; caught by _sustained_approach().
  - Passes:       ball crosses the frame with lateral velocity without
                  decelerating into the halo zone.
  - Walking:      causes motion but no sustained convergence on rim center.
  - Net sway:     frame-diff baseline stays low; only a passing ball drives
                  the diff above NET_DIFF_THRESHOLD.

DETECTION PATHS
  Primary:   state machine (IDLE→APPROACH→COMMITTED→WAIT_OUTCOME→COOLDOWN)
             driven by YOLO/tracker ball position.
  Fallback:  net-ROI pixel-diff spike + recent approach memory → MAKE.
             Works even when YOLO drops out entirely.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class ShotAttempt:
    id: int
    start_frame: int
    start_px: Tuple[int, int]
    result: str = "pending"
    end_frame: int = -1
    confidence: float = 0.0
    signal_breakdown: Dict[str, float] = field(default_factory=dict)


class _Phase(Enum):
    IDLE      = "idle"
    APPROACH  = "approach"   # ball moving toward rim from court side (deep shots)
    WATCHING  = "watching"   # ball inside halo from close range; watching silently
                             # no attempt is created yet; no miss can be logged
                             # resets quietly if ball exits or times out
    COMMITTED = "committed"  # ball inside halo after APPROACH (deep shots only)
                             # a miss CAN be logged here
    WAIT_OUT  = "wait_out"   # ball entered scoring zone; watching for outcome
    COOLDOWN  = "cooldown"   # outcome registered; suppressing re-triggers


# ─── Shot logic ───────────────────────────────────────────────────────────────

class ShotLogic:
    """
    Shot detector tuned for top-down backboard camera.

    Public API (same as original):
        update(ball_center, frame) -> Optional[ShotAttempt]
        debug_state()              -> Dict
        current                    -> Optional[ShotAttempt]
        frame_idx                  -> int
    """

    # ── Zone multipliers (× rim_radius_px) ───────────────────────────────────
    OUTER_SCALE   = 5.0    # beyond this → not in play
    HALO_SCALE    = 1.8    # "committed" zone
    SCORING_SCALE = 1.05   # inside the rim opening
    NET_ROI_SCALE = 0.72   # net interior circle for pixel-diff sampling

    # ── Approach quality ──────────────────────────────────────────────────────
    MIN_APPROACH_FRAMES  = 5     # frames the ball must be tracked converging
    MIN_DISTANCE_CLOSED  = 50    # px the ball must close toward rim
    CONVERGE_RATIO       = 0.60  # fraction of frames that must be converging
    MAX_APPROACH_DROPOUT = 12    # consecutive undetected frames before cancelling
    APPROACH_TIMEOUT     = 90    # total frames in APPROACH before cancelling

    # ── Make detection ────────────────────────────────────────────────────────
    # Pixel-diff threshold: mean absolute pixel change in net interior circle.
    # Baseline (still net):  ~2–5.   Ball through net: ~10–30+.
    NET_DIFF_THRESHOLD   = 10.0  # tune up if false positives from net sway
    NET_DIFF_SPIKE_HOLD  = 2     # consecutive frames above threshold to confirm
    DISAPPEAR_FRAMES     = 5     # frames ball absent inside scoring zone → make
    WAIT_OUT_TIMEOUT     = 35    # frames in WAIT_OUT before calling miss

    # ── Miss detection ────────────────────────────────────────────────────────
    REVERSAL_RATIO    = 0.60   # fraction of recent frames diverging = reversal
    REVERSAL_WINDOW   = 8
    COMMITTED_TIMEOUT = 50     # frames in COMMITTED without outcome → miss

    # ── Cooldown / suppression ────────────────────────────────────────────────
    COOLDOWN_FRAMES = 45

    # ── Fallback (net-diff only, no ball) ─────────────────────────────────────
    # If YOLO is dead but approach activity was recently seen, a net-diff spike
    # alone can trigger a make.
    APPROACH_MEMORY_FRAMES   = 90   # frames we remember "approach was active"
    FALLBACK_DIFF_MULTIPLIER = 1.4  # fallback needs higher diff to avoid false pos

    def __init__(
        self,
        rim_center: Tuple[int, int],
        rim_radius_px: int,
        # ── Tunable overrides (also accepts old param names silently) ─────────
        net_diff_threshold: float = NET_DIFF_THRESHOLD,
        score_cooldown_frames: int = COOLDOWN_FRAMES,
        **_,  # absorb legacy params (ignored)
    ) -> None:
        self.rim_center  = rim_center
        self.rim_radius  = rim_radius_px

        self.NET_DIFF_THRESHOLD = net_diff_threshold
        self.COOLDOWN_FRAMES    = score_cooldown_frames

        # Precomputed zone radii
        self._outer_r   = rim_radius_px * self.OUTER_SCALE
        self._halo_r    = rim_radius_px * self.HALO_SCALE
        self._scoring_r = rim_radius_px * self.SCORING_SCALE
        self._net_r     = int(rim_radius_px * self.NET_ROI_SCALE)

        # Persistent across resets
        self._frame_idx            = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._diff_buf: Deque[float]           = deque(maxlen=10)
        self._last_approach_frame: int         = -10_000
        self.attempt_id                        = 0

        self.current: Optional[ShotAttempt]   = None
        self._reset()

    # ─── Public API ───────────────────────────────────────────────────────────

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    def update(
        self,
        ball_center: Optional[Tuple[int, int]],
        frame: Optional[np.ndarray] = None,
    ) -> Optional[ShotAttempt]:
        """Call once per frame. Returns ShotAttempt when a shot resolves."""
        self._frame_idx += 1

        # Always compute net-ROI pixel diff (enables fallback even with no ball)
        net_diff = self._compute_net_diff(frame)
        self._diff_buf.append(net_diff)
        spike = self._diff_spike()

        dist = self._dist(ball_center)

        # Update approach memory whenever ball is seen outside the halo
        if ball_center is not None and dist is not None:
            if self._halo_r < dist < self._outer_r:
                self._last_approach_frame = self._frame_idx

        outcome = self._dispatch(ball_center, dist, spike)

        # Fallback: net-diff-only make detection (YOLO completely dead)
        if outcome is None and self._phase == _Phase.IDLE:
            frames_since_approach = self._frame_idx - self._last_approach_frame
            if (frames_since_approach < self.APPROACH_MEMORY_FRAMES
                    and net_diff >= self.NET_DIFF_THRESHOLD * self.FALLBACK_DIFF_MULTIPLIER):
                outcome = self._register("make")

        if frame is not None:
            self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return outcome

    def debug_state(self) -> Dict:
        last_diff = self._diff_buf[-1] if self._diff_buf else 0.0
        return {
            # Geometry (for overlay drawing)
            "rim_center":       self.rim_center,
            "scoring_radius":   int(self._scoring_r),
            "halo_radius":      int(self._halo_r),
            "net_roi_radius":   self._net_r,
            # Phase / signal state
            "phase":            self._phase.value,
            "net_diff":         round(last_diff, 1),
            "diff_spike":       self._diff_spike(),
            "approach_frames":  len(self._approach_dists),
            "absent_scoring":   self._absent_scoring,
            "flow_confirm":     self._flow_confirm,
            "has_active_attempt": self.current is not None,
            "attempt_id":       self.current.id if self.current else None,
            # Legacy keys expected by draw_debug_overlay in main.py
            "entry_line_y":        self.rim_center[1],
            "net_line_y":          self.rim_center[1],
            "net_lane_x_left":     self.rim_center[0] - self._net_r,
            "net_lane_x_right":    self.rim_center[0] + self._net_r,
            "inside_rim_frames":   self._absent_scoring,
            "net_lane_frames":     self._flow_confirm,
            "seen_above_rim":      False,
            "confidence_score":    round(last_diff, 1),
            "signals":             {"phase": self._phase.value, "net_diff": round(last_diff, 1)},
            "max_net_flow":        round(max(self._diff_buf, default=0), 1),
        }

    # ─── Phase dispatch ───────────────────────────────────────────────────────

    def _dispatch(
        self,
        ball_center: Optional[Tuple[int, int]],
        dist: Optional[float],
        spike: bool,
    ) -> Optional[ShotAttempt]:
        if self._phase == _Phase.IDLE:
            return self._idle(ball_center, dist)
        if self._phase == _Phase.APPROACH:
            return self._approach(ball_center, dist)
        if self._phase == _Phase.WATCHING:
            return self._watching(ball_center, dist)
        if self._phase == _Phase.COMMITTED:
            return self._committed(ball_center, dist, spike)
        if self._phase == _Phase.WAIT_OUT:
            return self._wait_out(ball_center, dist, spike)
        if self._phase == _Phase.COOLDOWN:
            if self._frame_idx - self._phase_start >= self.COOLDOWN_FRAMES:
                self._reset()
        return None

    # ── IDLE ─────────────────────────────────────────────────────────────────

    def _idle(
        self,
        ball_center: Optional[Tuple[int, int]],
        dist: Optional[float],
    ) -> Optional[ShotAttempt]:
        if ball_center is None or dist is None:
            return None
        # Ignore balls completely outside the outer zone (off screen or far away)
        if dist >= self._outer_r:
            return None

        self.attempt_id += 1
        self.current = ShotAttempt(
            id=self.attempt_id,
            start_frame=self._frame_idx,
            start_px=ball_center,
        )
        self._last_approach_frame = self._frame_idx

        if dist < self._halo_r:
            # ── Close shot / layup: ball inside halo on first detection ───────
            # Go to WATCHING — a silent observation phase.
            # No ShotAttempt is created yet and no miss can be logged.
            # We only commit once the ball enters the scoring zone, which
            # filters dribbles and passes that stay in the halo but never
            # actually go through the rim.
            self._watching_start = self._frame_idx
            self._watching_pos   = ball_center
            self._phase          = _Phase.WATCHING
            self._phase_start    = self._frame_idx
            return None
        else:
            # ── Normal shot: ball in outer zone, track approach ───────────────
            if not self._moving_toward_rim(ball_center):
                self._reset()
                return None
            self._approach_dists = [dist]
            self._approach_pos   = [ball_center]
            self._phase          = _Phase.APPROACH
            self._phase_start    = self._frame_idx
            self._dropout_frames = 0

        return None

    # ── APPROACH ─────────────────────────────────────────────────────────────

    def _approach(
        self,
        ball_center: Optional[Tuple[int, int]],
        dist: Optional[float],
    ) -> Optional[ShotAttempt]:
        elapsed = self._frame_idx - self._phase_start

        if ball_center is not None and dist is not None:
            self._dropout_frames = 0
            self._approach_dists.append(dist)
            self._approach_pos.append(ball_center)
            self._last_approach_frame = self._frame_idx

            # ── Entered halo → COMMITTED ──────────────────────────────────────
            if dist < self._halo_r:
                self._phase         = _Phase.COMMITTED
                self._phase_start   = self._frame_idx
                self._committed_abs = 0
                self._recent_dists  = deque(maxlen=self.REVERSAL_WINDOW)
                return None

            # ── Ball fled the shot zone → cancel ──────────────────────────────
            if dist > self._outer_r:
                self._reset()
                return None

            # ── Anti-dribble / anti-pass: verify sustained approach ───────────
            # Only check after enough history so we don't cancel too eagerly.
            if len(self._approach_dists) > 15 and not self._sustained_approach():
                self._reset()
                return None

        else:
            # Tracking dropout — allow a brief gap, then cancel
            self._dropout_frames += 1
            if self._approach_dists:
                self._approach_dists.append(self._approach_dists[-1])  # hold last dist
            if self._dropout_frames > self.MAX_APPROACH_DROPOUT:
                self._reset()
                return None

        if elapsed > self.APPROACH_TIMEOUT:
            self._reset()
        return None

    # ── WATCHING ─────────────────────────────────────────────────────────────

    # Max frames to watch without the ball entering the scoring zone before
    # quietly giving up. Keep short — a real shot reaches the zone quickly.
    WATCHING_TIMEOUT = 30

    def _watching(
        self,
        ball_center: Optional[Tuple[int, int]],
        dist: Optional[float],
    ) -> Optional[ShotAttempt]:
        """
        Silent observation phase for close-range shots.
        No ShotAttempt exists yet; no miss is ever logged here.
        We only escalate when the ball enters the scoring zone.
        Dribbles and passes through the halo zone fall through quietly.
        """
        elapsed = self._frame_idx - self._phase_start

        if ball_center is not None and dist is not None:
            # Ball entered the scoring zone → it's a real attempt
            if dist < self._scoring_r:
                self.attempt_id += 1
                self.current = ShotAttempt(
                    id=self.attempt_id,
                    start_frame=self._frame_idx,
                    start_px=ball_center,
                )
                self._last_approach_frame  = self._frame_idx
                self._phase                = _Phase.WAIT_OUT
                self._phase_start          = self._frame_idx
                self._absent_scoring       = 0
                self._absent_retreating    = 0
                self._last_in_scoring      = True
                self._flow_confirm         = 0
                return None

            # Ball exited halo entirely — not a shot, reset silently
            if dist > self._halo_r * 1.2:
                self._reset()
                return None

        # Timeout without entering scoring zone — dribble or pass, ignore
        if elapsed > self.WATCHING_TIMEOUT:
            self._reset()
        return None

    # ── COMMITTED ────────────────────────────────────────────────────────────

    def _committed(
        self,
        ball_center: Optional[Tuple[int, int]],
        dist: Optional[float],
        spike: bool,
    ) -> Optional[ShotAttempt]:
        elapsed = self._frame_idx - self._phase_start

        if ball_center is not None and dist is not None:
            self._recent_dists.append(dist)
            self._committed_abs = 0

            # ── Entered scoring zone → WAIT_OUT ───────────────────────────────
            if dist < self._scoring_r:
                self._phase             = _Phase.WAIT_OUT
                self._phase_start       = self._frame_idx
                self._absent_scoring    = 0
                self._absent_retreating = 0
                self._last_in_scoring   = True
                self._flow_confirm      = 0
                return None

            # ── Ball reversed sharply away from rim → MISS ────────────────────
            # Only call a miss if the ball is clearly outside the halo again AND
            # the recent trajectory is consistently diverging.
            if dist > self._halo_r * 1.3 and self._reversed():
                return self._register("miss")

        else:
            # Ball vanished inside the halo zone
            self._committed_abs += 1
            # Net spike + ball already absent = fell through quickly
            if spike and self._committed_abs >= 3:
                return self._register("make")

        if elapsed > self.COMMITTED_TIMEOUT:
            return self._register("miss")
        return None

    # ── WAIT_OUT ─────────────────────────────────────────────────────────────

    # Frames ball must be absent while retreating before calling miss.
    # A small buffer handles brief YOLO dropout while ball is still in halo.
    RETREATING_MISS_FRAMES = 3

    def _wait_out(
        self,
        ball_center: Optional[Tuple[int, int]],
        dist: Optional[float],
        spike: bool,
    ) -> Optional[ShotAttempt]:
        """
        Ball entered the rim interior. Confirm make or miss.

        The core distinction:
          MAKE — ball was last seen inside the scoring zone and then disappears
                 (it fell below the rim plane into the net, camera loses it)
          MISS — ball retreats to the halo zone and then disappears
                 (bounced upward/sideways out of the frame, never went through)

        We track _last_in_scoring to know which case applies when the ball
        goes absent, so a ball bouncing straight up out of frame is not
        mistaken for a make.
        """
        elapsed = self._frame_idx - self._phase_start

        if ball_center is not None and dist is not None:
            if dist > self._halo_r:
                # Ball reappeared well outside the rim — bounced back out → MISS
                return self._register("miss")

            if dist < self._scoring_r:
                # Ball still inside scoring zone
                self._last_in_scoring    = True
                self._absent_scoring     = 0
                self._absent_retreating  = 0
            else:
                # Ball in halo but outside scoring zone — retreating
                self._last_in_scoring    = False
                self._absent_retreating += 1

        else:
            # Ball absent — direction at disappearance determines outcome
            if self._last_in_scoring:
                # Disappeared from inside scoring zone → fell through net → MAKE candidate
                self._absent_scoring += 1
            else:
                # Disappeared while retreating (in halo) → bounced away → MISS
                self._absent_retreating += 1
                if self._absent_retreating >= self.RETREATING_MISS_FRAMES:
                    return self._register("miss")

        # Net pixel-diff spike only counts when ball has already disappeared
        # from inside the scoring zone. A spike while retreating just means
        # the ball was flying past the net ROI, not going through it.
        if spike and self._last_in_scoring and self._absent_scoring >= 1:
            self._flow_confirm += 1
            if self._flow_confirm >= self.NET_DIFF_SPIKE_HOLD:
                return self._register("make")
        else:
            self._flow_confirm = 0

        # Ball disappeared from inside scoring zone for N frames → MAKE
        if self._absent_scoring >= self.DISAPPEAR_FRAMES:
            return self._register("make")

        if elapsed > self.WAIT_OUT_TIMEOUT:
            return self._register("miss")
        return None

    # ─── Geometry helpers ─────────────────────────────────────────────────────

    def _dist(self, p: Optional[Tuple[int, int]]) -> Optional[float]:
        if p is None:
            return None
        rx, ry = self.rim_center
        return math.hypot(p[0] - rx, p[1] - ry)

    def _moving_toward_rim(self, ball_center: Tuple[int, int]) -> bool:
        """
        Returns True if the ball is moving toward the rim.
        Requires at least one prior approach position; if not, optimistically
        returns True (we'll verify via _sustained_approach later).
        """
        if not self._approach_pos:
            return True
        prev = self._approach_pos[-1]
        rx, ry = self.rim_center
        dx, dy = rx - prev[0], ry - prev[1]   # direction to rim
        mx, my = ball_center[0] - prev[0], ball_center[1] - prev[1]  # motion
        norm = math.hypot(dx, dy) or 1.0
        dot = (mx * dx + my * dy) / norm
        # Lenient: allow slightly sideways motion (corner shots etc.)
        return dot > -3.0

    def _sustained_approach(self) -> bool:
        """
        True if the recent approach history shows a sustained, consistent
        convergence toward the rim — not a dribble or random motion.
        """
        dists = self._approach_dists[-15:]
        if len(dists) < self.MIN_APPROACH_FRAMES:
            return True  # not enough history yet, give benefit of doubt
        converging = sum(
            1 for i in range(1, len(dists)) if dists[i] < dists[i - 1]
        )
        ratio     = converging / (len(dists) - 1)
        net_closed = dists[0] - dists[-1]
        return ratio >= self.CONVERGE_RATIO and net_closed >= self.MIN_DISTANCE_CLOSED

    def _reversed(self) -> bool:
        """True if recent distance trend is consistently moving AWAY from rim."""
        dists = list(self._recent_dists)
        if len(dists) < 4:
            return False
        diverging = sum(
            1 for i in range(1, len(dists)) if dists[i] > dists[i - 1]
        )
        return diverging / (len(dists) - 1) >= self.REVERSAL_RATIO

    # ─── Net-ROI pixel diff ───────────────────────────────────────────────────

    def _compute_net_diff(self, frame: Optional[np.ndarray]) -> float:
        """
        Mean absolute pixel change inside the net interior circle.

        We use frame-difference (absdiff) rather than Farneback optical flow:
          - Much faster on Jetson
          - Sufficient to detect the large pixel change caused by a ball
            passing through the white net
          - Baseline (stationary net) ≈ 2–5; ball through net ≈ 10–30+
        """
        if frame is None or self._prev_gray is None:
            return 0.0

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rx, ry = self.rim_center
        nr = self._net_r
        x1 = max(0, rx - nr)
        y1 = max(0, ry - nr)
        x2 = min(w, rx + nr)
        y2 = min(h, ry + nr)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        crop_cur = gray[y1:y2, x1:x2]
        crop_prv = self._prev_gray[y1:y2, x1:x2]

        diff = cv2.absdiff(crop_cur, crop_prv).astype(np.float32)

        # Mask to inscribed circle only (exclude corners of the bounding box)
        cy_off = ry - y1   # rim center in crop coords
        cx_off = rx - x1
        Y, X   = np.ogrid[:diff.shape[0], :diff.shape[1]]
        circle_mask = (X - cx_off) ** 2 + (Y - cy_off) ** 2 <= nr ** 2
        in_circle = diff[circle_mask]
        if in_circle.size == 0:
            return 0.0
        return float(np.mean(in_circle))

    def _diff_spike(self) -> bool:
        if not self._diff_buf:
            return False
        return self._diff_buf[-1] >= self.NET_DIFF_THRESHOLD

    # ─── Outcome registration ─────────────────────────────────────────────────

    def _register(self, result: str) -> ShotAttempt:
        """Finalise the current attempt and transition to COOLDOWN."""
        if self.current is None:
            # Fallback path: no attempt object was created (YOLO-dead make)
            self.attempt_id += 1
            self.current = ShotAttempt(
                id=self.attempt_id,
                start_frame=self._frame_idx,
                start_px=self.rim_center,
            )
        self.current.result    = result
        self.current.end_frame = self._frame_idx
        peak_diff = round(max(self._diff_buf, default=0.0), 1)
        self.current.confidence = peak_diff
        self.current.signal_breakdown = {
            "phase_at_outcome": self._phase.value,
            "peak_net_diff":    peak_diff,
            "approach_frames":  len(self._approach_dists),
            "absent_scoring":   self._absent_scoring,
        }
        done         = self.current
        self.current = None
        self._phase  = _Phase.COOLDOWN
        self._phase_start = self._frame_idx
        return done

    # ─── Internal reset ───────────────────────────────────────────────────────

    def _reset(self) -> None:
        """Reset per-attempt state. Does NOT reset frame_idx, flow buf, or session totals."""
        self._phase           = _Phase.IDLE
        self._phase_start     = -1
        self.current          = None
        self._approach_dists: List[float]            = []
        self._approach_pos:   List[Tuple[int, int]]  = []
        self._recent_dists:   Deque[float]           = deque(maxlen=self.REVERSAL_WINDOW)
        self._committed_abs   = 0
        self._absent_scoring  = 0
        self._flow_confirm    = 0
        self._dropout_frames      = 0
        self._watching_start      = -1
        self._watching_pos: Optional[Tuple[int, int]] = None
        self._last_in_scoring     = False
        self._absent_retreating   = 0
