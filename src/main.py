from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

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


def draw_debug_overlay(
    frame,
    debug: dict,
    rim_excl_radius: int,
    ball_center: Optional[Tuple[int, int]],
    foot_px: Optional[Tuple[int, int]],
    fps: float,
) -> None:
    rx, ry = debug["rim_center"]
    scoring_radius = debug["scoring_radius"]
    entry_y = debug["entry_line_y"]
    net_y = debug["net_line_y"]
    net_x1 = debug["net_lane_x_left"]
    net_x2 = debug["net_lane_x_right"]
    h, w = frame.shape[:2]

    # Scoring zone around rim center
    cv2.circle(frame, (rx, ry), scoring_radius, (0, 180, 255), 1)

    # Rim exclusion zone (detections inside here are suppressed)
    cv2.circle(frame, (rx, ry), rim_excl_radius, (0, 60, 220), 1)

    # Entry line: ball should be seen above this during an attempt
    cv2.line(frame, (0, max(0, entry_y)), (w - 1, max(0, entry_y)), (90, 180, 255), 1)

    # Net lane rectangle: downward pass through this confirms make
    top = max(0, net_y)
    cv2.rectangle(frame, (max(0, net_x1), top), (min(w - 1, net_x2), h - 1), (0, 220, 120), 1)

    # Tracked ball position
    if ball_center is not None:
        cv2.circle(frame, ball_center, 9, (0, 255, 255), 2)
        cv2.drawMarker(frame, ball_center, (0, 255, 255), cv2.MARKER_CROSS, 14, 1)

    # Player foot position used for shot location
    if foot_px is not None:
        cv2.drawMarker(frame, foot_px, (255, 80, 0), cv2.MARKER_TRIANGLE_UP, 18, 2)

    status = (
        f"A:{int(debug['has_active_attempt'])} "
        f"AR:{int(debug['seen_above_rim'])} "
        f"RIM:{debug['inside_rim_frames']} "
        f"NET:{debug['net_lane_frames']} "
        f"CONF:{debug['confidence_score']}"
    )
    cv2.putText(frame, status, (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 2)


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

    ok, frame = cam.read()
    if not ok:
        raise RuntimeError("Could not read first frame from camera")

    # --- Calibration ---
    rim_pt = pick_points("Calibrate Rim", frame, 1, "Click rim center")
    if len(rim_pt) != 1:
        raise RuntimeError("Rim calibration failed")
    rim_center = rim_pt[0]

    shot_logic = ShotLogic(
        rim_center=rim_center,
        rim_radius_px=h_cfg["rim_radius_px"],
        score_cooldown_frames=h_cfg["score_cooldown_frames"],
        make_confirm_frames=h_cfg["make_confirm_frames"],
        miss_timeout_frames=h_cfg["miss_timeout_frames"],
        min_shot_arc_drop_px=h_cfg["min_shot_arc_drop_px"],
        entry_above_margin_px=h_cfg.get("entry_above_margin_px", 8),
        net_drop_margin_px=h_cfg.get("net_drop_margin_px", 14),
        net_lane_radius_scale=h_cfg.get("net_lane_radius_scale", 0.8),
        net_confirm_frames=h_cfg.get("net_confirm_frames", 2),
        use_confidence_scoring=bool(h_cfg.get("use_confidence_scoring", True)),
        make_threshold=float(h_cfg.get("make_threshold", 60.0)),
        net_flow_threshold=float(h_cfg.get("net_flow_threshold", 1.5)),
        min_tracking_frames=int(h_cfg.get("min_tracking_frames", 10)),
        min_travel_px=float(h_cfg.get("min_travel_px", 150.0)),
        min_launch_velocity_px=float(h_cfg.get("min_launch_velocity_px", 5.0)),
        weights=h_cfg.get("weights"),
    )

    # Activate rim exclusion zone on the detector now that rim_center is known
    rim_radius_px = h_cfg["rim_radius_px"]
    excl_scale = float(det_cfg.get("rim_exclusion_scale", 1.5))
    detector.set_rim(rim_center, int(rim_radius_px * excl_scale))

    mapper = CourtMapper(
        court_width_ft=cfg.calibration["court_width_ft"],
        court_length_ft=cfg.calibration["court_length_ft"],
    )
    print("Click 4 court corners in this order: near-left baseline, near-right baseline, far-right, far-left")
    court_img_pts = pick_points("Calibrate Court", frame, 4, "Pick 4 court points")
    if len(court_img_pts) == 4:
        court_world_pts = [(0.0, 0.0), (50.0, 0.0), (50.0, 47.0), (0.0, 47.0)]
        mapper.set_homography(court_img_pts, court_world_pts)
    else:
        print("Court calibration skipped; shot chart will not include coordinates")

    store = SessionStore(cfg.output["session_dir"])

    # --- Main loop ---
    shot_start_foot_px: Optional[Tuple[int, int]] = None
    prev_has_attempt = False
    last_foot_px: Optional[Tuple[int, int]] = None  # most recent person detection for overlay
    frame_count = 0
    fps_start = time.time()
    live_fps = 0.0

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        frame_count += 1

        detections = detector.detect(frame)
        track = tracker.update(detections)
        center = track.center if track is not None else None

        done = shot_logic.update(center, frame)

        # Capture foot position at the moment an attempt is triggered
        has_attempt = shot_logic.current is not None
        if has_attempt and not prev_has_attempt:
            feet = detector.detect_person_feet()
            shot_start_foot_px = feet[0] if feet else None
        # Keep overlay marker fresh even between shots
        feet_now = detector.detect_person_feet()
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
