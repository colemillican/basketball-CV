from __future__ import annotations

import random
import threading
import time
from typing import Callable, Dict, Optional, Tuple

from camera import Camera
from config import load_config
from court_mapper import CourtMapper
from detector import build_detector
from session import SessionStore
from shot_chart import save_shot_chart
from shot_logic import ShotLogic
from tracker import BallTracker


class WorkoutRuntime:
    def __init__(self, config_path: str, on_event: Callable[[Dict], None]) -> None:
        self.cfg = load_config(config_path)
        self.on_event = on_event
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            raise RuntimeError("Session already running")
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _emit(self, event_type: str, **payload) -> None:
        self.on_event({"type": event_type, **payload})

    def _run(self) -> None:
        backend = str(self.cfg.raw.get("runtime", {}).get("backend", "real")).lower()
        if backend == "mock":
            self._run_mock()
            return
        if backend == "dual_benchmark":
            self._run_dual_benchmark()
            return
        if backend == "real_dual_skeleton":
            self._run_real_dual_skeleton()
            return
        self._run_real()

    def _finalize_session(self, store: Optional[SessionStore]) -> None:
        if store is None:
            return
        csv_path = store.save_csv(self.cfg.output["shots_csv"])
        summary_path = store.save_summary(self.cfg.output["summary_json"])
        chart_path = store.session_dir / self.cfg.output["chart_png"]
        save_shot_chart(store.records, chart_path)
        self._emit(
            "session_complete",
            running=False,
            output={
                "shots_csv": str(csv_path),
                "summary_json": str(summary_path),
                "shot_chart": str(chart_path),
            },
        )

    def _run_real(self) -> None:
        self._running = True
        cam = None
        store = None
        frame_count = 0

        try:
            cam = Camera(self.cfg.video["source"], self.cfg.video["width"], self.cfg.video["height"])
            detector = build_detector(self.cfg.raw.get("detector", {}))
            tracker = BallTracker(
                max_distance_px=self.cfg.tracking["max_distance_px"],
                max_missed_frames=self.cfg.tracking["max_missed_frames"],
            )

            ok, frame = cam.read()
            if not ok:
                raise RuntimeError("Could not read camera frame")

            runtime_cal = self.cfg.raw.get("runtime_calibration", {})
            rim_center = tuple(runtime_cal.get("rim_center_px", []))
            if len(rim_center) != 2:
                rim_center = (frame.shape[1] // 2, frame.shape[0] // 3)
                self._emit(
                    "warning",
                    message="runtime_calibration.rim_center_px missing; using fallback center estimate",
                )

            shot_logic = self._build_shot_logic(rim_center)

            mapper = CourtMapper(
                court_width_ft=self.cfg.calibration["court_width_ft"],
                court_length_ft=self.cfg.calibration["court_length_ft"],
            )

            court_pts = runtime_cal.get("court_image_points", [])
            if isinstance(court_pts, list) and len(court_pts) == 4:
                img_pts = [tuple(p) for p in court_pts]
                world_pts = [(0.0, 0.0), (50.0, 0.0), (50.0, 47.0), (0.0, 47.0)]
                mapper.set_homography(img_pts, world_pts)
            else:
                self._emit(
                    "warning",
                    message="runtime_calibration.court_image_points missing; shot chart coordinates disabled",
                )

            store = SessionStore(self.cfg.output["session_dir"])
            makes = 0
            total = 0
            self._emit("status", running=True, makes=makes, total=total, fg_percent=0.0)

            while not self._stop_evt.is_set():
                ok, frame = cam.read()
                if not ok:
                    self._emit("error", message="Camera stream ended")
                    break

                frame_count += 1
                detections = detector.detect(frame)
                track = tracker.update(detections)
                center = track.center if track is not None else None

                done = shot_logic.update(center, frame)
                if done is not None:
                    total += 1
                    if done.result == "make":
                        makes += 1
                    court_xy = mapper.map_pixel_to_court(done.start_px)
                    store.add_shot(
                        shot_id=done.id,
                        frame_start=done.start_frame,
                        frame_end=done.end_frame,
                        result=done.result,
                        start_px=done.start_px,
                        court_xy=court_xy,
                    )
                    fg = (makes / total * 100.0) if total else 0.0
                    self._emit(
                        "shot",
                        shot_id=done.id,
                        result=done.result,
                        start_px=done.start_px,
                        court_xy=court_xy,
                        makes=makes,
                        total=total,
                        fg_percent=round(fg, 1),
                    )

                if frame_count % 30 == 0:
                    fg = (makes / total * 100.0) if total else 0.0
                    self._emit("status", running=True, makes=makes, total=total, fg_percent=round(fg, 1))

                time.sleep(0.001)

            self._finalize_session(store)

        except Exception as exc:
            self._emit("error", message=str(exc))
        finally:
            if cam is not None:
                cam.release()
            self._running = False

    def _run_mock(self) -> None:
        self._running = True
        store = None

        try:
            mock_cfg = self.cfg.raw.get("runtime", {}).get("mock", {})
            shot_interval_sec = float(mock_cfg.get("shot_interval_sec", 2.2))
            make_probability = float(mock_cfg.get("make_probability", 0.62))
            jitter_ft = float(mock_cfg.get("location_jitter_ft", 2.0))

            zones = mock_cfg.get(
                "shot_zones_ft",
                [
                    [25.0, 8.0],   # paint
                    [15.0, 18.0],  # left elbow
                    [35.0, 18.0],  # right elbow
                    [8.0, 22.0],   # left wing
                    [42.0, 22.0],  # right wing
                    [5.0, 6.0],    # left corner
                    [45.0, 6.0],   # right corner
                    [25.0, 28.0],  # top key
                ],
            )

            store = SessionStore(self.cfg.output["session_dir"])
            makes = 0
            total = 0
            self._emit("status", running=True, makes=makes, total=total, fg_percent=0.0)
            self._emit("warning", message="Running in MOCK mode (no camera required)")

            next_shot_ts = time.time() + shot_interval_sec
            shot_id = 0

            while not self._stop_evt.is_set():
                now = time.time()
                if now >= next_shot_ts:
                    shot_id += 1
                    total += 1
                    result = "make" if random.random() <= make_probability else "miss"
                    if result == "make":
                        makes += 1

                    base_x, base_y = random.choice(zones)
                    cx = max(0.5, min(49.5, base_x + random.uniform(-jitter_ft, jitter_ft)))
                    cy = max(0.5, min(46.5, base_y + random.uniform(-jitter_ft, jitter_ft)))

                    # Mock pixel position only for event display parity.
                    start_px = (int(cx * 24 + 60), int(700 - cy * 12))
                    frame_start = shot_id * 30
                    frame_end = frame_start + 12

                    store.add_shot(
                        shot_id=shot_id,
                        frame_start=frame_start,
                        frame_end=frame_end,
                        result=result,
                        start_px=start_px,
                        court_xy=(cx, cy),
                    )
                    fg = (makes / total * 100.0) if total else 0.0
                    self._emit(
                        "shot",
                        shot_id=shot_id,
                        result=result,
                        start_px=start_px,
                        court_xy=(round(cx, 2), round(cy, 2)),
                        makes=makes,
                        total=total,
                        fg_percent=round(fg, 1),
                    )
                    next_shot_ts = now + shot_interval_sec
                else:
                    if total == 0 or total % 3 == 0:
                        fg = (makes / total * 100.0) if total else 0.0
                        self._emit("status", running=True, makes=makes, total=total, fg_percent=round(fg, 1))
                    time.sleep(0.2)

            self._finalize_session(store)

        except Exception as exc:
            self._emit("error", message=str(exc))
        finally:
            self._running = False

    def _camera_settings(self, prefix: str) -> Tuple[int | str, int, int]:
        v = self.cfg.video
        source = v.get(f"{prefix}_source", v["source"])
        width = int(v.get(f"{prefix}_width", v["width"]))
        height = int(v.get(f"{prefix}_height", v["height"]))
        return source, width, height

    def _build_shot_logic(self, rim_center: tuple) -> "ShotLogic":
        h = self.cfg.shot_logic
        return ShotLogic(
            rim_center=rim_center,
            rim_radius_px=h["rim_radius_px"],
            score_cooldown_frames=h["score_cooldown_frames"],
            make_confirm_frames=h["make_confirm_frames"],
            miss_timeout_frames=h["miss_timeout_frames"],
            min_shot_arc_drop_px=h["min_shot_arc_drop_px"],
            entry_above_margin_px=h.get("entry_above_margin_px", 8),
            net_drop_margin_px=h.get("net_drop_margin_px", 14),
            net_lane_radius_scale=h.get("net_lane_radius_scale", 0.8),
            net_confirm_frames=h.get("net_confirm_frames", 2),
            use_confidence_scoring=bool(h.get("use_confidence_scoring", True)),
            make_threshold=float(h.get("make_threshold", 60.0)),
            net_flow_threshold=float(h.get("net_flow_threshold", 1.5)),
            min_tracking_frames=int(h.get("min_tracking_frames", 10)),
            min_travel_px=float(h.get("min_travel_px", 150.0)),
            weights=h.get("weights"),
        )

    def _build_mapper_from_calibration(self, key: str) -> CourtMapper:
        mapper = CourtMapper(
            court_width_ft=self.cfg.calibration["court_width_ft"],
            court_length_ft=self.cfg.calibration["court_length_ft"],
        )
        runtime_cal = self.cfg.raw.get("runtime_calibration", {})
        court_pts = runtime_cal.get(key, runtime_cal.get("court_image_points", []))
        if isinstance(court_pts, list) and len(court_pts) == 4:
            img_pts = [tuple(p) for p in court_pts]
            world_pts = [(0.0, 0.0), (50.0, 0.0), (50.0, 47.0), (0.0, 47.0)]
            mapper.set_homography(img_pts, world_pts)
        else:
            self._emit("warning", message=f"{key} missing; coordinates disabled for this stream")
        return mapper

    def _run_dual_benchmark(self) -> None:
        self._running = True
        rim_cam = None
        wide_cam = None
        start_ts = time.time()

        try:
            cfg = self.cfg.raw.get("runtime", {}).get("dual_benchmark", {})
            duration_sec = float(cfg.get("duration_sec", 30))
            report_interval_sec = float(cfg.get("report_interval_sec", 1.0))
            rim_stride = max(1, int(cfg.get("rim_stride", 1)))
            wide_stride = max(1, int(cfg.get("wide_stride", 2)))
            use_detection = bool(cfg.get("use_detection", True))

            rim_source, rim_w, rim_h = self._camera_settings("rim")
            wide_source, wide_w, wide_h = self._camera_settings("wide")
            rim_cam = Camera(rim_source, rim_w, rim_h)
            wide_cam = Camera(wide_source, wide_w, wide_h)

            rim_detector = build_detector(self.cfg.raw.get("detector_rim", self.cfg.raw.get("detector", {})))
            wide_detector = build_detector(self.cfg.raw.get("detector_wide", self.cfg.raw.get("detector", {})))

            rim_read = wide_read = 0
            rim_infer = wide_infer = 0
            loop_idx = 0
            last_report = start_ts
            self._emit("warning", message="Running dual_benchmark mode")

            while not self._stop_evt.is_set():
                now = time.time()
                elapsed = now - start_ts
                if elapsed >= duration_sec:
                    break

                ok_rim, rim_frame = rim_cam.read()
                ok_wide, wide_frame = wide_cam.read()
                if ok_rim:
                    rim_read += 1
                if ok_wide:
                    wide_read += 1

                if use_detection:
                    if ok_rim and (loop_idx % rim_stride == 0):
                        _ = rim_detector.detect(rim_frame)
                        rim_infer += 1
                    if ok_wide and (loop_idx % wide_stride == 0):
                        _ = wide_detector.detect(wide_frame)
                        wide_infer += 1

                loop_idx += 1

                if now - last_report >= report_interval_sec:
                    rim_fps = rim_read / max(elapsed, 1e-6)
                    wide_fps = wide_read / max(elapsed, 1e-6)
                    rim_inf_fps = rim_infer / max(elapsed, 1e-6)
                    wide_inf_fps = wide_infer / max(elapsed, 1e-6)
                    self._emit(
                        "status",
                        running=True,
                        makes=0,
                        total=0,
                        fg_percent=0.0,
                    )
                    self._emit(
                        "warning",
                        message=(
                            f"dual_benchmark: rim_read={rim_fps:.1f}fps wide_read={wide_fps:.1f}fps "
                            f"rim_infer={rim_inf_fps:.1f}fps wide_infer={wide_inf_fps:.1f}fps"
                        ),
                    )
                    last_report = now

            elapsed = max(time.time() - start_ts, 1e-6)
            summary = {
                "elapsed_sec": round(elapsed, 2),
                "rim_read_fps": round(rim_read / elapsed, 2),
                "wide_read_fps": round(wide_read / elapsed, 2),
                "rim_infer_fps": round(rim_infer / elapsed, 2),
                "wide_infer_fps": round(wide_infer / elapsed, 2),
                "rim_frames": rim_read,
                "wide_frames": wide_read,
            }
            self._emit("warning", message=f"dual_benchmark complete: {summary}")
            self._emit(
                "session_complete",
                running=False,
                output={"benchmark": summary},
            )
        except Exception as exc:
            self._emit("error", message=str(exc))
        finally:
            if rim_cam is not None:
                rim_cam.release()
            if wide_cam is not None:
                wide_cam.release()
            self._running = False

    def _run_real_dual_skeleton(self) -> None:
        """
        Two-stream skeleton:
        - Rim stream: make/miss event detection
        - Wide stream: location proxy from latest wide tracked ball center
        This is a structural scaffold for profiling/integration, not final fusion logic.
        """
        self._running = True
        rim_cam = None
        wide_cam = None
        store = None

        try:
            rim_source, rim_w, rim_h = self._camera_settings("rim")
            wide_source, wide_w, wide_h = self._camera_settings("wide")
            rim_cam = Camera(rim_source, rim_w, rim_h)
            wide_cam = Camera(wide_source, wide_w, wide_h)

            rim_detector = build_detector(self.cfg.raw.get("detector_rim", self.cfg.raw.get("detector", {})))
            wide_detector = build_detector(self.cfg.raw.get("detector_wide", self.cfg.raw.get("detector", {})))
            rim_tracker = BallTracker(
                max_distance_px=self.cfg.tracking["max_distance_px"],
                max_missed_frames=self.cfg.tracking["max_missed_frames"],
            )
            wide_tracker = BallTracker(
                max_distance_px=self.cfg.tracking["max_distance_px"],
                max_missed_frames=self.cfg.tracking["max_missed_frames"],
            )

            ok_rim, rim_frame = rim_cam.read()
            if not ok_rim:
                raise RuntimeError("Could not read first rim frame")

            runtime_cal = self.cfg.raw.get("runtime_calibration", {})
            rim_center = tuple(runtime_cal.get("rim_center_px", []))
            if len(rim_center) != 2:
                rim_center = (rim_frame.shape[1] // 2, rim_frame.shape[0] // 3)
                self._emit("warning", message="rim_center_px missing; using fallback")

            shot_logic = self._build_shot_logic(rim_center)

            # Skeleton choice: location mapping uses wide stream calibration.
            wide_mapper = self._build_mapper_from_calibration("wide_court_image_points")
            store = SessionStore(self.cfg.output["session_dir"])
            makes = total = 0
            self._emit(
                "warning",
                message="Running real_dual_skeleton mode (wide location is latest-center proxy, not release-synced)",
            )
            self._emit("status", running=True, makes=0, total=0, fg_percent=0.0)

            last_wide_center: Optional[Tuple[int, int]] = None
            frame_idx = 0

            while not self._stop_evt.is_set():
                ok_rim, rim_frame = rim_cam.read()
                ok_wide, wide_frame = wide_cam.read()
                if not ok_rim or not ok_wide:
                    self._emit("error", message="Dual stream read failed")
                    break

                frame_idx += 1
                rim_track = rim_tracker.update(rim_detector.detect(rim_frame))
                wide_track = wide_tracker.update(wide_detector.detect(wide_frame))
                rim_center_px = rim_track.center if rim_track is not None else None
                if wide_track is not None:
                    last_wide_center = wide_track.center

                done = shot_logic.update(rim_center_px, rim_frame)
                if done is not None:
                    total += 1
                    if done.result == "make":
                        makes += 1

                    # Proxy mapping from latest wide center. Replace with release-synced mapping later.
                    court_xy = wide_mapper.map_pixel_to_court(last_wide_center) if last_wide_center is not None else None
                    store.add_shot(
                        shot_id=done.id,
                        frame_start=done.start_frame,
                        frame_end=done.end_frame,
                        result=done.result,
                        start_px=done.start_px,
                        court_xy=court_xy,
                    )
                    fg = (makes / total * 100.0) if total else 0.0
                    self._emit(
                        "shot",
                        shot_id=done.id,
                        result=done.result,
                        start_px=done.start_px,
                        court_xy=court_xy,
                        makes=makes,
                        total=total,
                        fg_percent=round(fg, 1),
                    )

                if frame_idx % 30 == 0:
                    fg = (makes / total * 100.0) if total else 0.0
                    self._emit("status", running=True, makes=makes, total=total, fg_percent=round(fg, 1))

                time.sleep(0.001)

            self._finalize_session(store)
        except Exception as exc:
            self._emit("error", message=str(exc))
        finally:
            if rim_cam is not None:
                rim_cam.release()
            if wide_cam is not None:
                wide_cam.release()
            self._running = False
