"""
Export a trained YOLOv8 .pt model to TensorRT .engine format.

MUST be run on the Jetson — TensorRT engines are device-specific and cannot
be built on one machine and deployed on another.

Usage:
    python tools/export_engine.py models/basketball_topdown.pt

Takes 5-10 minutes on Jetson Orin Nano. Output is saved alongside the .pt file.
"""
import argparse
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLOv8 .pt → TensorRT .engine")
    p.add_argument("model", help="Path to trained .pt weights file")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Input image size used during training (default: 640)")
    p.add_argument("--fp32", action="store_true",
                   help="Use FP32 instead of FP16. Slower but may be more accurate.")
    return p.parse_args()


def main():
    args = parse_args()
    pt_path = Path(args.model)

    if not pt_path.exists():
        raise SystemExit(f"ERROR: {pt_path} not found")
    if pt_path.suffix != ".pt":
        raise SystemExit(f"ERROR: expected a .pt file, got {pt_path.suffix}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed — run: pip install ultralytics")

    half = not args.fp32
    precision = "FP16" if half else "FP32"
    engine_path = pt_path.with_suffix(".engine")

    print(f"Model:     {pt_path}")
    print(f"Precision: {precision}")
    print(f"Image sz:  {args.imgsz}")
    print(f"Output:    {engine_path}")
    print()
    print("Building TensorRT engine — this takes 5-10 minutes on Jetson Orin Nano...")
    print()

    start = time.time()

    model = YOLO(str(pt_path))
    model.export(
        format="engine",
        imgsz=args.imgsz,
        half=half,
        device=0,
    )

    elapsed = time.time() - start

    if engine_path.exists():
        size_mb = engine_path.stat().st_size / 1_048_576
        print(f"\nExport complete in {elapsed:.0f}s")
        print(f"Engine:    {engine_path}  ({size_mb:.1f} MB)")
        print()
        print("Update configs/default.yaml:")
        print(f"  detector:")
        print(f"    model_path: {engine_path}")
        print(f"    ball_class_ids: [0]   # custom single-class model, not COCO class 32")
        print(f"    confidence: 0.45      # raise from 0.25 — custom model is more precise")
    else:
        print(f"\nERROR: export finished but {engine_path} was not created")
        print("Check ultralytics and TensorRT versions are compatible with JetPack 6.1")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
