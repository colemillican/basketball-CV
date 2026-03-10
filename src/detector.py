from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


class BaseBallDetector:
    def detect(self, frame) -> List[Tuple[int, int, int]]:
        raise NotImplementedError


class OrangeBallDetector(BaseBallDetector):
    """
    HSV-based detector for a basketball-like orange color. For a production gym setup,
    replace this with a learned detector and keep the same return format.
    """

    def __init__(self) -> None:
        self.lower = np.array([5, 80, 80], dtype=np.uint8)
        self.upper = np.array([25, 255, 255], dtype=np.uint8)

    def detect(self, frame) -> List[Tuple[int, int, int]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
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
            detections.append((int(x), int(y), int(radius)))

        detections.sort(key=lambda d: d[2], reverse=True)
        return detections[:3]


class YoloBallDetector(BaseBallDetector):
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        ball_class_ids: Sequence[int] | None = None,
        max_detections: int = 3,
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
        self.ball_class_ids = set(ball_class_ids or [32])  # COCO sports ball
        self.max_detections = max_detections

    def detect(self, frame) -> List[Tuple[int, int, int]]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )
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
            detections.append((cx, cy, radius))

        detections.sort(key=lambda d: d[2], reverse=True)
        return detections[: self.max_detections]


def build_detector(cfg: Dict) -> BaseBallDetector:
    backend = str(cfg.get("backend", "yolo")).lower()
    if backend == "yolo":
        return YoloBallDetector(
            model_path=cfg.get("model_path", "yolov8n.pt"),
            confidence=float(cfg.get("confidence", 0.25)),
            ball_class_ids=cfg.get("ball_class_ids", [32]),
            max_detections=int(cfg.get("max_detections", 3)),
        )
    if backend == "orange":
        return OrangeBallDetector()
    raise ValueError(f"Unknown detector backend: {backend}")
