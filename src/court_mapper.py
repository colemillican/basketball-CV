from typing import List, Optional, Tuple

import cv2
import numpy as np


class CourtMapper:
    def __init__(self, court_width_ft: float = 50.0, court_length_ft: float = 94.0) -> None:
        self.court_width_ft = court_width_ft
        self.court_length_ft = court_length_ft
        self.H: Optional[np.ndarray] = None

    def set_homography(self, image_points: List[Tuple[int, int]], court_points_ft: List[Tuple[float, float]]) -> None:
        if len(image_points) < 4 or len(court_points_ft) < 4:
            raise ValueError("Need at least 4 point pairs for homography")
        src = np.array(image_points, dtype=np.float32)
        dst = np.array(court_points_ft, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        if H is None:
            raise RuntimeError("Could not estimate homography")
        self.H = H

    def map_pixel_to_court(self, p: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        if self.H is None:
            return None
        pt = np.array([[[p[0], p[1]]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(mapped[0]), float(mapped[1])
