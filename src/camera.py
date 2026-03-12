from typing import Tuple

import cv2


class Camera:
    def __init__(self, source: int | str, width: int, height: int, fps: int = 60) -> None:
        self.cap = cv2.VideoCapture(source)
        # Request MJPEG from the camera before setting resolution/fps.
        # MJPEG compresses each frame before USB transmission, cutting bandwidth
        # ~10x vs YUYV and unlocking higher framerates on USB cameras.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()

    def frame_size(self) -> Tuple[int, int]:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def actual_fps(self) -> float:
        """Query the fps the driver actually negotiated (may differ from requested)."""
        return self.cap.get(cv2.CAP_PROP_FPS)
