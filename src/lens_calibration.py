"""
Lens distortion calibration for wide-angle cameras (e.g. 2.1mm fisheye-style).

Usage:
    python src/main.py --calibrate-lens

Requires a printed checkerboard. Default is 9x6 inner corners (10x7 squares).
Print one from: https://calib.io/pages/camera-calibration-pattern-generator
    - Pattern type: Checkerboard
    - Rows: 7, Columns: 10 (gives 6x9 inner corners)
    - Square size: 30mm works well

Once calibrated, coefficients are saved to configs/lens_calibration.npz and
loaded automatically on every subsequent run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

def run_calibration(
    cam,
    save_path: Path,
    board_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 30.0,
    target_captures: int = 20,
) -> bool:
    """
    Interactive calibration routine. Shows live feed, auto-captures frames
    when the checkerboard is detected. Computes and saves coefficients when
    enough captures are collected.

    Returns True on success, False if cancelled.
    """
    cols, rows = board_size
    obj_point = np.zeros((rows * cols, 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    obj_points = []   # 3D world points
    img_points = []   # 2D image points
    frame_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    win = "Lens Calibration  |  Space=capture  Esc=cancel"
    cv2.namedWindow(win)

    print(f"\n=== Lens Calibration ===")
    print(f"Board: {cols}x{rows} inner corners, {square_size_mm}mm squares")
    print(f"Need {target_captures} captures. Move the board to different positions/angles.")
    print(f"Space = manual capture | 'a' = auto-capture mode | Esc = cancel\n")

    auto_capture = False
    last_auto_frame = -30

    frame_idx = 0
    while len(obj_points) < target_captures:
        ok, frame = cam.read()
        if not ok:
            break
        frame_idx += 1
        frame_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        display = frame.copy()
        status_color = (0, 255, 0) if found else (0, 100, 255)

        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, (cols, rows), corners2, found)

            do_capture = False
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                do_capture = True
            elif key == ord("a"):
                auto_capture = not auto_capture
                print(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")
            elif key == 27:
                cv2.destroyWindow(win)
                return False

            if auto_capture and (frame_idx - last_auto_frame) >= 30:
                do_capture = True

            if do_capture:
                obj_points.append(obj_point)
                img_points.append(corners2)
                last_auto_frame = frame_idx
                print(f"  Captured {len(obj_points)}/{target_captures}")
                status_color = (255, 255, 0)
        else:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyWindow(win)
                return False
            if key == ord("a"):
                auto_capture = not auto_capture
                print(f"Auto-capture: {'ON' if auto_capture else 'OFF'}")

        cv2.putText(display, f"Captures: {len(obj_points)}/{target_captures}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(display, f"Board {'FOUND' if found else 'searching...'}",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        mode = "AUTO" if auto_capture else "MANUAL"
        cv2.putText(display, f"Mode: {mode}  |  'a'=toggle  Space=capture  Esc=cancel",
                    (20, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow(win, display)

    cv2.destroyWindow(win)

    if len(obj_points) < 5:
        print("Not enough captures for calibration.")
        return False

    print(f"\nComputing calibration from {len(obj_points)} captures...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, frame_size, None, None
    )
    print(f"Reprojection error: {ret:.4f}px  (< 1.0 is good, < 0.5 is excellent)")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        frame_size=np.array(frame_size),
        reprojection_error=np.array([ret]),
    )
    print(f"Calibration saved to {save_path}\n")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Runtime undistortion
# ──────────────────────────────────────────────────────────────────────────────

class LensUndistorter:
    """
    Precomputes remap tables from saved calibration and applies undistortion
    per frame via cv2.remap (fast — avoids recomputing per frame).
    """

    def __init__(self, calibration_path: Path, frame_size: Tuple[int, int]) -> None:
        data = np.load(calibration_path)
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        err = float(data["reprojection_error"][0])

        # Optimal new camera matrix — alpha=0 crops to valid pixels only
        h, w = frame_size[1], frame_size[0]
        new_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha=0
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_matrix, (w, h), cv2.CV_16SC2
        )
        self._roi = roi
        print(f"Lens calibration loaded (reprojection error: {err:.4f}px)")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)


def load_undistorter(
    calibration_path: Path,
    frame_size: Tuple[int, int],
) -> Optional[LensUndistorter]:
    """Returns a LensUndistorter if calibration file exists, else None."""
    if not calibration_path.exists():
        return None
    return LensUndistorter(calibration_path, frame_size)
