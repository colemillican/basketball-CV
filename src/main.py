from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

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


def draw_debug_overlay(frame, debug: dict) -> None:
    rx, ry = debug["rim_center"]
    scoring_radius = debug["scoring_radius"]
    entry_y = debug["entry_line_y"]
    net_y = debug["net_line_y"]
    net_x1 = debug["net_lane_x_left"]
    net_x2 = debug["net_lane_x_right"]
    h, w = frame.shape[:2]

    # Scoring zone around rim center.
    cv2.circle(frame, (rx, ry), scoring_radius, (0, 180, 255), 1)

    # Entry line: ball should be seen above this at least once during attempt.
    cv2.line(frame, (0, max(0, entry_y)), (w - 1, max(0, entry_y)), (90, 180, 255), 1)

    # Net lane rectangle: downward pass through this lane confirms make.
    top = max(0, net_y)
    cv2.rectangle(frame, (max(0, net_x1), top), (min(w - 1, net_x2), h - 1), (0, 220, 120), 1)

    status = (
        f"A:{int(debug['has_active_attempt'])} "
        f"AR:{int(debug['seen_above_rim'])} "
        f"RIM:{debug['inside_rim_frames']} "
        f"NET:{debug['net_lane_frames']}"
    )
    cv2.putText(frame, status, (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    debug_overlay = bool(cfg.raw.get("debug", {}).get("show_overlay", True))

    cam = Camera(cfg.video["source"], cfg.video["width"], cfg.video["height"])
    detector = build_detector(cfg.raw.get("detector", {}))
    tracker = BallTracker(
        max_distance_px=cfg.tracking["max_distance_px"],
        max_missed_frames=cfg.tracking["max_missed_frames"],
    )

    ok, frame = cam.read()
    if not ok:
        raise RuntimeError("Could not read first frame from camera")

    rim_pt = pick_points("Calibrate Rim", frame, 1, "Click rim center")
    if len(rim_pt) != 1:
        raise RuntimeError("Rim calibration failed")
    rim_center = rim_pt[0]

    h_cfg = cfg.shot_logic
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
        weights=h_cfg.get("weights"),
    )

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

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        detections = detector.detect(frame)
        track = tracker.update(detections)
        center = track.center if track is not None else None

        attempt_done = shot_logic.update(center, frame)
        if attempt_done:
            court_xy = mapper.map_pixel_to_court(attempt_done.start_px)
            store.add_shot(
                shot_id=attempt_done.id,
                frame_start=attempt_done.start_frame,
                frame_end=attempt_done.end_frame,
                result=attempt_done.result,
                start_px=attempt_done.start_px,
                court_xy=court_xy,
            )
            print(f"Shot {attempt_done.id}: {attempt_done.result} @ px={attempt_done.start_px}, court={court_xy}")

        if center is not None:
            cv2.circle(frame, center, 9, (0, 255, 255), 2)
        cv2.circle(frame, rim_center, h_cfg["rim_radius_px"], (255, 120, 0), 2)
        if debug_overlay:
            draw_debug_overlay(frame, shot_logic.debug_state())

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
