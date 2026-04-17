from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from camera import Camera
from config import load_config
from court_mapper import CourtMapper
from detector import build_detector
from session import SessionStore
from shot_chart import save_shot_chart
from shot_logic import ShotLogic
from tracker import BallTracker


def pick_points(window_name: str, frame, count: int, prompt: str) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    temp = frame.copy()

    def on_mouse(event, x, y, _flags, _param):
        nonlocal temp
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < count:
            pts.append((x, y))
            cv2.circle(temp, (x, y), 5, (0, 255, 255), -1)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while len(pts) < count:
        preview = temp.copy()
        cv2.putText(preview, prompt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2)
        cv2.putText(preview, f"{len(pts)}/{count} selected", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window_name, preview)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyWindow(window_name)
    return pts


def pick_labeled_points(window_name: str, frame, labels: List[str]) -> List[Tuple[int, int]]:
    """Like pick_points but shows each label as the current target in the window."""
    pts: List[Tuple[int, int]] = []
    total = len(labels)
    temp = frame.copy()

    def on_mouse(event, x, y, _flags, _param):
        nonlocal temp
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < total:
            pts.append((x, y))
            cv2.circle(temp, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(temp, str(len(pts)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    for i, label in enumerate(labels):
        print(f"  [{i+1}/{total}] Click: {label}")

    while len(pts) < total:
        preview = temp.copy()
        idx = len(pts)
        label = labels[idx] if idx < total else ""
        cv2.putText(preview, f"[{idx+1}/{total}] {label}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 255, 40), 2)
        cv2.imshow(window_name, preview)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyWindow(window_name)
    return pts


def draw_debug_overlay(
    frame,
    debug: dict,
    rim_excl_radius: int,
    ball_center: Optional[Tuple[int, int]],
    ball_radius: int,
    foot_px: Optional[Tuple[int, int]],
    fps: float,
) -> None:
    rx, ry = debug["rim_center"]
    h, w = frame.shape[:2]

    # Scoring zone (inside rim)
    cv2.circle(frame, (rx, ry), debug["scoring_radius"], (0, 180, 255), 1)

    # Halo zone (committed threshold)
    cv2.circle(frame, (rx, ry), debug["halo_radius"], (0, 60, 220), 1)

    # Net ROI circle (pixel-diff sampling region)
    cv2.circle(frame, (rx, ry), debug["net_roi_radius"], (0, 220, 120), 1)

    # Tracked ball position
    if ball_center is not None:
        cv2.circle(frame, ball_center, max(ball_radius, 9), (0, 255, 255), 2)
        cv2.drawMarker(frame, ball_center, (0, 255, 255), cv2.MARKER_CROSS, 14, 1)

    # Player foot position
    if foot_px is not None:
        cv2.drawMarker(frame, foot_px, (255, 80, 0), cv2.MARKER_TRIANGLE_UP, 18, 2)

    phase   = debug.get("phase", "?")
    diff    = debug.get("net_diff", 0.0)
    spike   = debug.get("diff_spike", False)
    app_f   = debug.get("approach_frames", 0)
    absent  = debug.get("absent_scoring", 0)

    # Phase colour: green = active, yellow = committed/wait, grey = idle/cooldown
    phase_color = (180, 180, 180)
    if phase == "approach":
        phase_color = (50, 200, 50)
    elif phase == "watching":
        phase_color = (0, 180, 180)   # teal — observing, no commitment yet
    elif phase in ("committed", "wait_out"):
        phase_color = (0, 200, 255)

    cv2.putText(frame, f"PHASE: {phase.upper()}", (18, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, phase_color, 2)
    status = (
        f"DIFF:{diff:.1f}{'*' if spike else ' '} "
        f"APP:{app_f} "
        f"ABS:{absent} "
        f"ATT:{int(debug['has_active_attempt'])}"
    )
    cv2.putText(frame, status, (18, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 2)


class DetectorWorker:
    """Runs ball detection in a background thread.

    The main loop posts frames via post_frame() and reads the most recent
    detections via get_detections(). YOLO never blocks the main loop.
    """

    YOLO_W, YOLO_H = 640, 360  # downsample before YOLO to cut preprocessing cost

    def __init__(self, detector) -> None:
        self._detector = detector
        self._lock = threading.Lock()
        self._frame = None
        self._full_w = 1280
        self._full_h = 720
        self._detections: List[Tuple[int, int, int]] = []
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def post_frame(self, frame) -> None:
        with self._lock:
            self._frame = frame
            self._full_h, self._full_w = frame.shape[:2]
        self._new_frame.set()

    def get_detections(self) -> List[Tuple[int, int, int]]:
        with self._lock:
            return list(self._detections)

    def get_person_feet(self) -> List[Tuple[int, int]]:
        return self._detector.detect_person_feet()

    def stop(self) -> None:
        self._stop.set()
        self._new_frame.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._new_frame.wait()
            self._new_frame.clear()
            if self._stop.is_set():
                break

            with self._lock:
                frame = self._frame
                full_w = self._full_w
                full_h = self._full_h

            if frame is None:
                continue

            small = cv2.resize(frame, (self.YOLO_W, self.YOLO_H))
            raw = self._detector.detect(small)
            sx = full_w / self.YOLO_W
            sy = full_h / self.YOLO_H
            scaled = [(int(x * sx), int(y * sy), int(r * sx)) for x, y, r in raw]

            with self._lock:
                self._detections = scaled


def fit_circle(points: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], int]:
    """Least-squares circle fit from N boundary points.
    Returns (center_xy, radius_px)."""
    pts = np.array(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx = result[0] / 2
    cy = result[1] / 2
    r = float(np.sqrt(result[2] + cx ** 2 + cy ** 2))
    return (int(cx), int(cy)), int(r)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    debug_overlay = bool(cfg.raw.get("debug", {}).get("show_overlay", True))
    h_cfg = cfg.shot_logic
    det_cfg = cfg.raw.get("detector", {})

    cam = Camera(
        cfg.video["source"],
        cfg.video["width"],
        cfg.video["height"],
        fps=int(cfg.video.get("fps", 60)),
    )
    detector = build_detector(det_cfg)
    tracker = BallTracker(
        max_distance_px=cfg.tracking["max_distance_px"],
        max_missed_frames=cfg.tracking["max_missed_frames"],
    )

    # Warm up camera — first frames from USB cameras are often black
    print("Warming up camera...")
    frame = None
    for _ in range(60):
        ok, f = cam.read()
        if ok:
            frame = f
    if frame is None:
        raise RuntimeError("Could not read first frame from camera")

    # --- Calibration ---
    while True:
        rim_pts = pick_points("Calibrate Rim", frame, 5,
                              "Click 5 points on the rim circle (spread around it)")
        if len(rim_pts) < 3:
            raise RuntimeError("Need at least 3 rim points for calibration")
        rim_center, rim_radius_px = fit_circle(rim_pts)
        print(f"Rim fitted — center: {rim_center}, radius: {rim_radius_px}px")

        # Show fitted circle for confirmation
        confirm = frame.copy()
        cv2.circle(confirm, rim_center, rim_radius_px, (0, 255, 0), 2)
        cv2.circle(confirm, rim_center, 6, (0, 255, 0), -1)
        cv2.putText(confirm, f"center={rim_center}  radius={rim_radius_px}px",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(confirm, "SPACE = confirm    R = redo",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.namedWindow("Confirm Rim", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Confirm Rim", 854, 480)
        cv2.imshow("Confirm Rim", confirm)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("Confirm Rim")
        if key != ord("r"):
            break

    shot_logic = ShotLogic(
        rim_center=rim_center,
        rim_radius_px=rim_radius_px,
        net_diff_threshold=float(h_cfg.get("net_diff_threshold", 10.0)),
        score_cooldown_frames=int(h_cfg.get("score_cooldown_frames", 45)),
    )

    # Activate rim exclusion zone on the detector now that rim_center is known
    excl_scale = float(det_cfg.get("rim_exclusion_scale", 1.5))
    detector.set_rim(rim_center, int(rim_radius_px * excl_scale))

    mapper = CourtMapper(
        court_width_ft=cfg.calibration["court_width_ft"],
        court_length_ft=cfg.calibration["court_length_ft"],
    )
    ref_pts = cfg.calibration.get("reference_points", [])
    if len(ref_pts) >= 4:
        labels = [p["label"] for p in ref_pts]
        court_world_pts = [tuple(p["world"]) for p in ref_pts]
        court_img_pts = pick_labeled_points("Calibrate Court", frame, labels)
        if len(court_img_pts) == len(court_world_pts):
            mapper.set_homography(court_img_pts, court_world_pts)
        else:
            print("Court calibration skipped; shot chart will not include coordinates")
    else:
        print("calibration.reference_points not configured; court mapping disabled")

    store = SessionStore(cfg.output["session_dir"])

    # --- Threaded detector ---
    worker = DetectorWorker(detector)

    cv2.namedWindow("Basketball CV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Basketball CV", 854, 480)

    # --- Main loop ---
    shot_start_foot_px: Optional[Tuple[int, int]] = None
    prev_has_attempt = False
    last_foot_px: Optional[Tuple[int, int]] = None
    frame_count = 0
    fps_start = time.time()
    live_fps = 0.0

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        frame_count += 1

        # Post frame to background YOLO thread; get latest available detections
        worker.post_frame(frame)
        detections = worker.get_detections()

        track = tracker.update(detections)
        center = track.center if track is not None else None
        radius = track.radius if track is not None else None

        done = shot_logic.update(center, frame, ball_radius=radius)

        # Capture foot position at the moment an attempt is triggered
        has_attempt = shot_logic.current is not None
        if has_attempt and not prev_has_attempt:
            feet = worker.get_person_feet()
            shot_start_foot_px = feet[0] if feet else None
        feet_now = worker.get_person_feet()
        if feet_now:
            last_foot_px = feet_now[0]
        prev_has_attempt = has_attempt

        if done is not None:
            location_px = shot_start_foot_px if shot_start_foot_px is not None else done.start_px
            shot_start_foot_px = None
            court_xy = mapper.map_pixel_to_court(location_px)
            store.add_shot(
                shot_id=done.id,
                frame_start=done.start_frame,
                frame_end=done.end_frame,
                result=done.result,
                start_px=location_px,
                court_xy=court_xy,
            )
            print(f"Shot {done.id}: {done.result} conf={done.confidence} @ px={location_px}, court={court_xy}")

        # FPS calculation (rolling over last 30 frames)
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start
            live_fps = 30.0 / elapsed if elapsed > 0 else 0.0
            fps_start = time.time()

        if debug_overlay:
            draw_debug_overlay(
                frame,
                shot_logic.debug_state(),
                rim_excl_radius=int(rim_radius_px * excl_scale),
                ball_center=center,
                ball_radius=track.radius if track is not None else 9,
                foot_px=last_foot_px,
                fps=live_fps,
            )

        makes = sum(1 for r in store.records if r.result == "make")
        total = len(store.records)
        fg = (makes / total * 100.0) if total else 0.0
        cv2.putText(frame, f"Shots: {total}  Makes: {makes}  FG%: {fg:.1f}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
        cv2.putText(frame, "Press Q to end workout", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Basketball CV", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    worker.stop()
    cam.release()
    cv2.destroyAllWindows()

    csv_path = store.save_csv(cfg.output["shots_csv"])
    summary_path = store.save_summary(cfg.output["summary_json"])
    chart_path = store.session_dir / cfg.output["chart_png"]
    save_shot_chart(store.records, chart_path)

    print("Session complete")
    print(f"Shots CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Shot chart: {chart_path}")


if __name__ == "__main__":
    main()
