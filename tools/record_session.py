"""
Record raw footage for dataset collection.

Run this during a real shooting session. Captures from /dev/video0 at 1280x720
MJPEG and writes an AVI at ~30fps (every other frame from the 60fps stream).

Usage:
    python tools/record_session.py [--duration 600] [--output data/raw_video]

Tips:
    - Record at least 3 sessions covering different shooters, lighting, and shot zones
    - Include both makes and misses
    - 10-20 minutes total is enough for a solid dataset
    - Press Ctrl+C or let --duration expire to stop
"""
import argparse
import time
from pathlib import Path

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Record camera footage for dataset collection")
    p.add_argument("--duration", type=int, default=0,
                   help="Max recording duration in seconds (0 = unlimited, stop with Ctrl+C)")
    p.add_argument("--output", type=str, default="data/raw_video",
                   help="Output directory for recorded video files")
    p.add_argument("--source", type=int, default=0,
                   help="V4L2 camera device index (default: 0)")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"session_{timestamp}.avi"

    cap = cv2.VideoCapture(args.source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera device {args.source}")
        return 1

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        30,
        (actual_w, actual_h),
    )

    if not writer.isOpened():
        print(f"ERROR: Could not open video writer at {out_path}")
        cap.release()
        return 1

    print(f"Recording to {out_path}")
    if args.duration:
        print(f"Will stop after {args.duration}s — or press Ctrl+C")
    else:
        print("Press Ctrl+C to stop")
    print()

    frame_count = 0
    written_count = 0
    start_time = time.time()
    last_report = start_time

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed — stopping")
                break

            frame_count += 1

            # Write every other frame: 60fps capture → ~30fps output.
            # Keeps file size manageable while preserving ball motion clarity.
            if frame_count % 2 == 0:
                writer.write(frame)
                written_count += 1

            now = time.time()
            elapsed = now - start_time

            if now - last_report >= 10.0:
                print(f"  {elapsed:.0f}s elapsed — {written_count} frames written "
                      f"({written_count / 30:.0f}s of video)")
                last_report = now

            if args.duration and elapsed >= args.duration:
                print(f"\nDuration limit reached ({args.duration}s)")
                break

    except KeyboardInterrupt:
        print("\nStopped by user")

    cap.release()
    writer.release()

    duration_s = written_count / 30
    size_mb = out_path.stat().st_size / 1_048_576 if out_path.exists() else 0
    print(f"\nSaved {written_count} frames ({duration_s:.1f}s) to {out_path}")
    print(f"File size: {size_mb:.1f} MB")
    print(f"\nNext step: python tools/extract_frames.py {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
