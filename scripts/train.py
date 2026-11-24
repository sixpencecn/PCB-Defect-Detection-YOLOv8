"""YOLOv8 CPU 迁移训练脚本。

用法：
    python scripts/train.py --epochs 40 --batch 8 --imgsz 640
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import (CLASS_NAMES, DATASET, DEFAULT_IMGSZ, METRICS_PATH,
                    MODEL_DIR, PRETRAINED_PATH, ROOT)
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 PCB defect training on CPU")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--name", type=str, default=MODEL_DIR.name)
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision")
    parser.add_argument("--no-val", action="store_true", help="skip per-epoch validation")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    weights = str(PRETRAINED_PATH) if PRETRAINED_PATH.exists() else "yolov8n.pt"
    model = YOLO(weights)

    print(f"Training YOLOv8n on CPU | epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} "
          f"threads={args.threads} mosaic={args.mosaic}")
    results = model.train(
        data=str(DATASET / "data.yaml"),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device="cpu",
        workers=0,
        lr0=args.lr0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2.0,
        patience=args.patience,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=args.mosaic,
        project=str(MODEL_DIR.parent),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=True,
        amp=args.amp,
        val=not args.no_val,
        deterministic=False,
    )

    metrics = model.val(
        data=str(DATASET / "data.yaml"),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=0,
        verbose=False,
        plots=True,
    )
    result_dict = dict(metrics.results_dict)
    summary = {
        "model": "YOLOv8n",
        "device": "CPU",
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "classes": CLASS_NAMES,
        "test_metrics": {k: float(v) for k, v in result_dict.items() if isinstance(v, (int, float))},
        "best_weights": str(MODEL_DIR / "weights" / "best.pt"),
    }
    METRICS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Training summary:", json.dumps(summary["test_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
