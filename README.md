# Basketball CV (MVP)

Live-camera computer vision prototype for basketball workouts:
- tracks the ball
- detects shot attempts and classifies each as `make` or `miss`
- maps each shot to court coordinates (after calibration)
- saves a workout shot chart and stats at the end
- includes a local TV kiosk UI (`login -> mode -> live results`)

## What this MVP does
- Uses one fixed camera feed (`webcam` by default).
- Uses configurable ball detection:
  - `yolo` backend (default)
  - `orange` HSV baseline backend
- Uses manual calibration:
  - click rim center
  - click 4 court points to build homography
- Logs each shot with:
  - make/miss
  - frame range
  - image pixel launch location
  - mapped court location in feet
- Writes outputs to a timestamped folder in `output/`.

## Project layout
- `src/main.py`: app entry, calibration flow, live loop.
- `src/ui_server.py`: local kiosk web server + APIs + WebSocket updates.
- `src/runtime.py`: background workout runtime used by the UI server.
- `src/detector.py`: detector backends + detector factory.
- `src/tracker.py`: simple single-ball tracker.
- `src/shot_logic.py`: make/miss state machine.
- `src/court_mapper.py`: homography mapping pixel -> court feet.
- `src/session.py`: shot/session persistence.
- `src/shot_chart.py`: matplotlib half-court shot chart.
- `configs/default.yaml`: runtime config.

## Install
```bash
pip install -r requirements.txt
```

If WebSocket live updates do not connect, reinstall dependencies in your active environment to ensure `websockets` is present.

## Run
```bash
python src/main.py --config configs/default.yaml
```

## Run TV Kiosk UI (recommended MVP path)
```bash
uvicorn src.ui_server:app --host 0.0.0.0 --port 8000
```
Then open `http://<jetson-ip>:8000` on the TV browser.

## Run Without Hardware (Mock Mode)
The default config is set to mock runtime:
- `runtime.backend: mock`

When you start a session in the UI, simulated shot events are generated and full output files are saved.

## Dual-Stream Benchmark (2 camera profiling)
Use this to measure whether your device can keep up with two streams:
1. Set `runtime.backend: dual_benchmark` in `configs/default.yaml`
2. Set `video.rim_source` and `video.wide_source`
3. Start UI server and start a session from the UI

Benchmark summary is emitted in the UI messages and session output payload.

## Dual-Stream Skeleton Mode
For early integration testing (not final fusion quality):
1. Set `runtime.backend: real_dual_skeleton`
2. Configure:
   - `video.rim_source` / `video.wide_source`
   - `runtime_calibration.rim_center_px`
   - `runtime_calibration.wide_court_image_points`
3. Start session from UI

Current dual skeleton behavior:
- make/miss from rim stream
- shot location from latest wide-stream tracked center (proxy)

## Detector config
`configs/default.yaml` includes:
- `detector.backend`: `yolo` or `orange`
- `detector.model_path`: YOLO weights path (`yolov8n.pt` default)
- `detector.confidence`: confidence threshold
- `detector.ball_class_ids`: class IDs to treat as ball
  - COCO sports ball is `32`
  - Many custom single-class models use `0`

## UI config
`configs/default.yaml` also includes:
- `ui.players`: dropdown list shown on login screen
- `ui.modes`: workout modes shown before session start
- `runtime_calibration`: fixed rim/court calibration points used by UI runtime
- `runtime_calibration.wide_court_image_points`: wide stream homography points for dual mode
- `runtime.backend`: `mock` or `real`
  - `mock`: no camera needed, simulated shots for UI testing
  - `real`: uses camera + detector pipeline
- additional `runtime.backend` options:
  - `dual_benchmark`: two-stream performance benchmark
  - `real_dual_skeleton`: two-stream scaffold for integration testing
- `debug.show_overlay`: draws tuning overlays in `src/main.py` OpenCV view

## Calibration flow
1. First frame appears.
2. Click rim center once.
3. Click 4 court points in order:
   - near-left baseline corner
   - near-right baseline corner
   - far-right corner
   - far-left corner
4. Session starts.
5. Press `Q` to end session.

## Output files
Saved under `output/<timestamp>/`:
- `shots.csv`
- `summary.json`
- `shot_chart.png`

## Notes and limitations
- Best accuracy comes from a fine-tuned gym-specific YOLO model, not generic weights.
- Make/miss logic is heuristic and tuned for fixed camera placement (now requires above-rim approach + rim contact + downward net-lane pass).
- For production quality, add learned rim detection and sequence-based make/miss classification.
