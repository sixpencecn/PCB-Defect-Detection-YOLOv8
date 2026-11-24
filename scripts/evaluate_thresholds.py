"""置信度 / IOU 阈值对比实验，结果写入 CSV 供论文分析。

用法：python scripts/evaluate_thresholds.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DATASET, MODEL_PATH, OUTPUT_DIR
from ultralytics import YOLO

CONFIDENCES = [0.25, 0.35, 0.5, 0.65]
IOUS = [0.35, 0.45, 0.60]


def main():
    model = YOLO(str(MODEL_PATH))
    rows = []
    for conf in CONFIDENCES:
        for iou in IOUS:
            metrics = model.val(
                data=str(DATASET / "data.yaml"),
                split="test",
                conf=conf,
                iou=iou,
                imgsz=320,
                batch=16,
                device="cpu",
                workers=0,
                verbose=False,
            )
            result = metrics.results_dict
            rows.append({
                "conf": conf,
                "iou": iou,
                "precision": round(float(result["metrics/precision(B)"]), 5),
                "recall": round(float(result["metrics/recall(B)"]), 5),
                "mAP50": round(float(result["metrics/mAP50(B)"]), 5),
                "mAP50-95": round(float(result["metrics/mAP50-95(B)"]), 5),
            })
            print(f"conf={conf:.2f} iou={iou:.2f} -> "
                  f"P={rows[-1]['precision']:.4f} R={rows[-1]['recall']:.4f} "
                  f"mAP50={rows[-1]['mAP50']:.4f}")

    output = OUTPUT_DIR / "threshold_comparison.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Threshold comparison saved: {output}")


if __name__ == "__main__":
    main()
