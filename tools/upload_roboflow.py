"""
Upload extracted frames to Roboflow for labeling.

Usage:
    python tools/upload_roboflow.py --api-key YOUR_KEY
    python tools/upload_roboflow.py --api-key YOUR_KEY --frames data/frames/session_20260312_230420
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Upload frames to Roboflow project")
    p.add_argument("--api-key", required=True, help="Roboflow private API key")
    p.add_argument("--frames", default="data/frames/session_20260312_230420",
                   help="Directory of JPEG frames to upload")
    p.add_argument("--project", default="sniper-project", help="Roboflow project name")
    p.add_argument("--workspace", default=None,
                   help="Roboflow workspace slug (default: auto-detect from API key)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed.")
        print("Run: pip install roboflow")
        return 1

    frames_dir = Path(args.frames)
    if not frames_dir.exists():
        print(f"ERROR: frames directory not found: {frames_dir}")
        return 1

    images = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not images:
        print(f"ERROR: no images found in {frames_dir}")
        return 1

    print(f"Found {len(images)} images in {frames_dir}")
    print("Connecting to Roboflow...")

    rf = Roboflow(api_key=args.api_key)
    workspace = rf.workspace(args.workspace) if args.workspace else rf.workspace()
    project = workspace.project(args.project)

    print(f"Uploading to project '{args.project}'...\n")

    success = 0
    failed = 0
    for i, img_path in enumerate(images, 1):
        try:
            project.upload(str(img_path))
            success += 1
            if i % 50 == 0 or i == len(images):
                print(f"  {i}/{len(images)} uploaded ({failed} failed)")
        except Exception as e:
            failed += 1
            print(f"  WARN: failed to upload {img_path.name}: {e}")

    print(f"\nDone. {success} uploaded, {failed} failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
