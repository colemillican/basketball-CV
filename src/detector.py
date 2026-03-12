from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class BaseBallDetector:
    def detect(self, frame) -> List[Tuple[int, int, int]]:
        raise NotImplementedError

    def detect_person_feet(self) -> List[Tuple[int, int]]:
        """Return (cx, foot_y) for each detected person, largest first.
        Uses the cached result from the most recent detect() call — no extra inference.
        Base implementation returns empty list for non-YOLO detectors."""
        return []

    def set_rim(self, center: Tuple[int, int], exclusion_radius: int) -> None:
        """Configure the rim exclusion zone after calibration. Called by runtime."""


class OrangeBallDetector(BaseBallDetector):
    """
    HSV-based detector for a basketball-like orange color.

    Improvements over the original:
    - Background subtraction (MOG2) motion mask: stationary objects (rim, floor
      markings) are absorbed into the background and suppressed.
    - Rim exclusion zone: any detection whose center falls within
      exclusion_radius pixels of the known rim center is discarded.
    """

    def __init__(self, use_motion_mask: bool = True) -> None:
        self.lower = np.array([5, 80, 80], dtype=np.uint8)
        self.upper = np.array([25, 255, 255], dtype=np.uint8)
        self._rim_center: Optional[Tuple[int, int]] = None
        self._rim_excl_radius: int = 0
        self._bg_sub = (
            cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)
            if use_motion_mask
            else None
        )
        self._fg_mask: Optional[np.ndarray] = None

    def set_rim(self, center: Tuple[int, int], exclusion_radius: int) -> None:
        self._rim_center = center
        self._rim_excl_radius = exclusion_radius

    def detect(self, frame) -> List[Tuple[int, int, int]]:
        if self._bg_sub is not None:
            raw = self._bg_sub.apply(frame)
            self._fg_mask = cv2.dilate(raw, None, iterations=2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Restrict to moving regions only
        if self._fg_mask is not None:
            mask = cv2.bitwise_and(mask, self._fg_mask)

        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Tuple[int, int, int]] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 120:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < 6 or radius > 60:
                continue
            if self._in_rim_zone(int(x), int(y)):
                continue
            detections.append((int(x), int(y), int(radius)))

        detections.sort(key=lambda d: d[2], reverse=True)
        return detections[:3]

    def _in_rim_zone(self, x: int, y: int) -> bool:
        if self._rim_center is None or self._rim_excl_radius <= 0:
            return False
        rx, ry = self._rim_center
        return ((x - rx) ** 2 + (y - ry) ** 2) ** 0.5 <= self._rim_excl_radius


class YoloBallDetector(BaseBallDetector):
    """
    YOLO-based ball detector.

    Improvements over the original:
    - Background subtraction (MOG2) motion mask: each detection bbox must
      contain sufficient foreground pixels, eliminating stationary false
      positives (e.g. the rim being detected as a sports ball).
    - Rim exclusion zone: detections centered on the known rim location
      are discarded.
    - motion_overlap_threshold: mean foreground pixel value that a bbox must
      exceed to be accepted (0-255 scale; default 10 means ~4% coverage).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.45,
        ball_class_ids: Sequence[int] | None = None,
        max_detections: int = 3,
        use_motion_mask: bool = True,
        motion_overlap_threshold: float = 10.0,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLO detection. "
                "Install with: pip install ultralytics"
            ) from exc

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.ball_class_ids = set(ball_class_ids or [32])
        self.max_detections = max_detections
        self.motion_overlap_threshold = motion_overlap_threshold

        self._rim_center: Optional[Tuple[int, int]] = None
        self._rim_excl_radius: int = 0
        self._bg_sub = (
            cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=False)
            if use_motion_mask
            else None
        )
        self._fg_mask: Optional[np.ndarray] = None
        self._last_results = None  # cached per-frame; reused by detect_person_feet()

    def set_rim(self, center: Tuple[int, int], exclusion_radius: int) -> None:
        self._rim_center = center
        self._rim_excl_radius = exclusion_radius

    def detect(self, frame) -> List[Tuple[int, int, int]]:
        if self._bg_sub is not None:
            raw = self._bg_sub.apply(frame)
            self._fg_mask = cv2.dilate(raw, None, iterations=2)

        self._last_results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )
        results = self._last_results
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None:
            return []

        detections: List[Tuple[int, int, int]] = []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            if cls[i] not in self.ball_class_ids:
                continue
            x1, y1, x2, y2 = xyxy[i]
            cx = int((x1 + x2) * 0.5)
            cy = int((y1 + y2) * 0.5)
            radius = int(max(x2 - x1, y2 - y1) * 0.5)
            if radius < 2:
                continue
            if self._in_rim_zone(cx, cy):
                continue
            if not self._has_motion(x1, y1, x2, y2):
                continue
            detections.append((cx, cy, radius))

        detections.sort(key=lambda d: d[2], reverse=True)
        return detections[: self.max_detections]

    def detect_person_feet(self) -> List[Tuple[int, int]]:
        """Return (cx, foot_y) for each detected person, sorted by bbox area descending.
        Reuses the cached result from the last detect() call — no extra YOLO inference."""
        if self._last_results is None:
            return []
        boxes = self._last_results[0].boxes
        if boxes is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        persons: List[Tuple[int, int, float]] = []
        for i in range(len(xyxy)):
            if cls[i] != 0:  # COCO person class
                continue
            x1, y1, x2, y2 = xyxy[i]
            cx = int((x1 + x2) * 0.5)
            foot_y = int(y2)  # bottom of bbox ≈ feet on floor
            area = (x2 - x1) * (y2 - y1)
            persons.append((cx, foot_y, area))
        persons.sort(key=lambda p: p[2], reverse=True)
        return [(p[0], p[1]) for p in persons]

    def _in_rim_zone(self, x: int, y: int) -> bool:
        if self._rim_center is None or self._rim_excl_radius <= 0:
            return False
        rx, ry = self._rim_center
        return ((x - rx) ** 2 + (y - ry) ** 2) ** 0.5 <= self._rim_excl_radius

    def _has_motion(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Return True if the detection bbox overlaps significantly with the motion mask."""
        if self._fg_mask is None:
            # BGS still warming up — allow all detections through
            return True
        roi = self._fg_mask[int(y1) : int(y2), int(x1) : int(x2)]
        if roi.size == 0:
            return False
        return float(np.mean(roi)) > self.motion_overlap_threshold


def build_detector(cfg: Dict) -> BaseBallDetector:
    backend = str(cfg.get("backend", "yolo")).lower()
    use_motion_mask = bool(cfg.get("use_motion_mask", True))
    if backend == "yolo":
        return YoloBallDetector(
            model_path=cfg.get("model_path", "yolov8n.pt"),
            confidence=float(cfg.get("confidence", 0.45)),
            ball_class_ids=cfg.get("ball_class_ids", [32]),
            max_detections=int(cfg.get("max_detections", 3)),
            use_motion_mask=use_motion_mask,
            motion_overlap_threshold=float(cfg.get("motion_overlap_threshold", 10.0)),
        )
    if backend == "orange":
        return OrangeBallDetector(use_motion_mask=use_motion_mask)
    raise ValueError(f"Unknown detector backend: {backend}")
