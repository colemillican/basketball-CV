"""
Fine-tune YOLOv8n for top-down basketball detection.

Run this on Google Colab (free T4 GPU) or any machine with an NVIDIA GPU.
Do NOT run on the Jetson — training is too slow there.

--- Google Colab setup ---
1. Runtime → Change runtime type → T4 GPU
2. Upload your Roboflow dataset ZIP (basketball-topdown.zip) to Colab
3. Paste or upload this script, then run:
       !python train_colab.py --data basketball-topdown.zip

--- Local GPU setup ---
    pip install ultralytics
    python tools/train_colab.py --data /path/to/basketball-topdown.zip

Output:
    runs/detect/basketball_topdown_<n>/weights/best.pt
    Download best.pt and copy it to models/ on the Jetson.
"""
import argparse
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8n on basketball dataset")
    p.add_argument("--data", required=True,
                   help="Path to Roboflow export ZIP or extracted data.yaml")
    p.add_argument("--epochs", type=int, default=100,
                   help="Training epochs (default: 100; use 50 for a quick test run)")
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size (default: 16; reduce to 8 if GPU OOM)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Input image size (default: 640)")
    p.add_argument("--base-model", type=str, default="yolov8n.pt",
                   help="Starting weights (default: yolov8n.pt — COCO pretrained)")
    return p.parse_args()


def find_data_yaml(data_arg: str) -> Path:
    path = Path(data_arg)

    # If a ZIP was provided, extract it first
    if path.suffix == ".zip":
        extract_dir = path.parent / path.stem
        print(f"Extracting {path} → {extract_dir}/")
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(extract_dir)
        path = extract_dir

    # Find data.yaml inside the extracted folder
    if path.is_dir():
        candidates = list(path.rglob("data.yaml"))
        if not candidates:
            raise FileNotFoundError(f"No data.yaml found inside {path}")
        return candidates[0]

    if path.is_file() and path.name == "data.yaml":
        return path

    raise FileNotFoundError(f"Could not locate data.yaml from: {data_arg}")


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed — run: pip install ultralytics")

    data_yaml = find_data_yaml(args.data)
    print(f"Dataset:    {data_yaml}")
    print(f"Base model: {args.base_model}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print()

    model = YOLO(args.base_model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="basketball_topdown",

        # ── Early stopping ─────────────────────────────────────────────────────
        patience=20,        # stop if mAP50 doesn't improve for 20 epochs

        # ── Augmentation tuned for top-down gym footage ────────────────────────
        # Color: mild — don't destroy the orange hue that helps detection
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        # Geometry: moderate scale variance — ball is very large (close to rim)
        # or small (far shot at arc peak), so scale aug is important
        scale=0.4,
        translate=0.1,
        degrees=10,         # camera may shift slightly between sessions
        # Flip: horizontal only — vertical flip would put the rim above the ball,
        # which never happens in your camera geometry
        fliplr=0.5,
        flipud=0.0,
        # Mosaic helps with detecting the ball at varied context/scales
        mosaic=0.5,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights: {best}")
    print()

    # Print mAP summary
    metrics = results.results_dict
    map50 = metrics.get("metrics/mAP50(B)", 0)
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0)
    print(f"mAP50:      {map50:.3f}  (target: >0.85)")
    print(f"mAP50-95:   {map50_95:.3f}")

    if map50 >= 0.85:
        print("Detection quality: GOOD — ready to deploy")
    elif map50 >= 0.75:
        print("Detection quality: ACCEPTABLE — worth testing on the Jetson")
    else:
        print("Detection quality: LOW — consider labeling more frames and re-training")

    print()
    print("Next steps:")
    print(f"  1. Download {best}")
    print(f"  2. Copy to Jetson:  scp best.pt <user>@<jetson-ip>:~/basketball-CV/models/basketball_topdown.pt")
    print(f"  3. On Jetson:       python tools/export_engine.py models/basketball_topdown.pt")
    print(f"  4. Update config:   model_path: models/basketball_topdown.engine")
    print(f"                      ball_class_ids: [0]")
    print(f"                      confidence: 0.45")


if __name__ == "__main__":
    main()
