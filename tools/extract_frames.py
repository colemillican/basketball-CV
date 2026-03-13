"""
Extract frames from recorded video for labeling.

Samples every Nth frame to avoid redundant near-duplicates while keeping
enough variety across shot arcs, player positions, and lighting.

Usage:
    # Extract from one video
    python tools/extract_frames.py data/raw_video/session_20260312_143000.avi

    # Extract from all videos in a directory
    python tools/extract_frames.py data/raw_video/

    # Custom sample rate (default: every 6th frame ≈ 5fps from 30fps video)
    python tools/extract_frames.py data/raw_video/ --every 4

Target: 800-1500 frames total across all sessions.
Output goes to data/frames/<video_stem>/ as numbered JPEGs.
"""
import argparse
import sys
from pathlib import Path

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Extract frames from recorded sessions for labeling")
    p.add_argument("source", help="Video file or directory containing video files")
    p.add_argument("--every", type=int, default=6,
                   help="Sample every Nth frame (default: 6, ~5fps from 30fps video)")
    p.add_argument("--output", type=str, default="data/frames",
                   help="Root output directory (default: data/frames)")
    p.add_argument("--quality", type=int, default=92,
                   help="JPEG quality 0-100 (default: 92)")
    return p.parse_args()


def extract_video(video_path: Path, out_root: Path, every: int, quality: int) -> int:
    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: could not open {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"  {video_path.name}: reading frames (every {every})...")

    saved = 0
    i = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % every == 0:
            out_path = out_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(out_path), frame, encode_params)
            saved += 1
        i += 1

    duration_s = i / fps
    print(f"    {i} total frames ({duration_s:.0f}s @ {fps:.0f}fps)")

    cap.release()
    print(f"    → {saved} frames saved to {out_dir}/")
    return saved


def main():
    args = parse_args()
    source = Path(args.source)
    out_root = Path(args.output)

    if not source.exists():
        print(f"ERROR: {source} does not exist")
        return 1

    if source.is_file():
        videos = [source]
    elif source.is_dir():
        videos = sorted(source.glob("*.avi")) + sorted(source.glob("*.mp4"))
        if not videos:
            print(f"No .avi or .mp4 files found in {source}")
            return 1
    else:
        print(f"ERROR: {source} is not a file or directory")
        return 1

    print(f"Extracting frames (every {args.every}) from {len(videos)} video(s):\n")

    total_saved = 0
    for video in videos:
        total_saved += extract_video(video, out_root, args.every, args.quality)

    print(f"\nTotal frames extracted: {total_saved}")

    if total_saved < 500:
        print(f"WARNING: {total_saved} frames is on the low side. "
              f"Consider recording more sessions or lowering --every.")
    elif total_saved > 3000:
        print(f"NOTE: {total_saved} frames is large. "
              f"Consider raising --every to reduce labeling work.")
    else:
        print("Frame count looks good for labeling.")

    print(f"\nNext step: upload {out_root}/ to Roboflow for labeling")
    print("  1. roboflow.com → New Project → Object Detection")
    print("  2. Upload frames, use Auto-Label, correct mistakes")
    print("  3. One class only: 'ball'")
    print("  4. Export as YOLOv8 format → download ZIP")
    print("  5. Train: python tools/train_colab.py  (or use the Colab notebook)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
