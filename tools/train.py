"""Fine-tune the detector (YOLO11n) and the crop classifier (YOLO11n-cls) on CPU.

Both start from the COCO / ImageNet weights published by Ultralytics. The best
checkpoints are copied to models/detector.pt and models/classifier.pt.

Usage:
    python tools/train.py detector --epochs 25
    python tools/train.py classifier --epochs 30
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def train_detector(args) -> Path:
    from ultralytics import YOLO

    model = YOLO(args.weights or "yolo11n.pt")
    model.train(
        data=str(Path(args.data or "data/datasets/detect/data.yaml").resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz or 640,
        batch=args.batch or 8,
        workers=1,
        device="cpu",
        patience=8,
        project=str(Path("runs").resolve()),
        name=args.name or "detector",
        exist_ok=True,
        seed=0,
        deterministic=False,
        cache=False,  # 8 GB of RAM: read from disk
        degrees=10,
        flipud=0.5,  # top-down camera: up/down has no meaning
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=3,
        plots=True,
        verbose=True,
    )
    best = Path("runs") / (args.name or "detector") / "weights" / "best.pt"
    Path("models").mkdir(exist_ok=True)
    shutil.copy2(best, "models/detector.pt")
    print("saved models/detector.pt")
    return best


def train_classifier(args) -> Path:
    from ultralytics import YOLO

    model = YOLO(args.weights or "yolo11n-cls.pt")
    model.train(
        data=str(Path(args.data or "data/datasets/classify").resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz or 96,
        batch=args.batch or 64,
        workers=1,
        device="cpu",
        patience=8,
        project=str(Path("runs").resolve()),
        name=args.name or "classifier",
        exist_ok=True,
        seed=0,
        degrees=180,
        flipud=0.5,
        fliplr=0.5,
        plots=True,
    )
    best = Path("runs") / (args.name or "classifier") / "weights" / "best.pt"
    Path("models").mkdir(exist_ok=True)
    shutil.copy2(best, "models/classifier.pt")
    print("saved models/classifier.pt")
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["detector", "classifier"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--data")
    ap.add_argument("--weights")
    ap.add_argument("--name")
    args = ap.parse_args(argv)
    (train_detector if args.what == "detector" else train_classifier)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
