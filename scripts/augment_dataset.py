"""训练集离线数据增强：随机翻转、亮度扰动、高斯噪声、缩放。

对每张训练图生成 4 个增强样本，并同步变换 YOLO 标注框。
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DATASET

SEED = 2026
VARIANTS = 4


def read_labels(path):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) == 5:
            boxes.append(tuple(float(v) for v in values))
    return boxes


def write_labels(path, boxes):
    path.write_text(
        "\n".join(" ".join(f"{v:.6f}" if i else str(int(v)) for i, v in enumerate(box)) for box in boxes) + "\n",
        encoding="utf-8",
    )


def flip_boxes(boxes):
    return [(c, 1.0 - cx, cy, w, h) for c, cx, cy, w, h in boxes]


def zoom_boxes(boxes, scale):
    offset = (scale - 1.0) / 2.0
    result = []
    for c, cx, cy, w, h in boxes:
        nx, ny = (cx - offset) * scale, (cy - offset) * scale
        nw, nh = w * scale, h * scale
        if 0 < nx < 1 and 0 < ny < 1 and nw > 0 and nh > 0:
            result.append((c, min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0), min(nw, 1.0), min(nh, 1.0)))
    return result


def main():
    rng = random.Random(SEED)
    image_dir = DATASET / "images" / "train"
    label_dir = DATASET / "labels" / "train"
    images = sorted(image_dir.glob("*.jpg"))
    generated = 0
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        boxes = read_labels(label_dir / f"{image_path.stem}.txt")
        height, width = image.shape[:2]

        # 1) 水平翻转
        variants = [(cv2.flip(image, 1), flip_boxes(boxes))]

        # 2) 亮度/对比度扰动
        alpha, beta = rng.uniform(1.08, 1.28), rng.uniform(8, 22)
        variants.append((cv2.convertScaleAbs(image, alpha=alpha, beta=beta), boxes))

        # 3) 高斯噪声
        sigma = rng.uniform(5, 11)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        variants.append((noisy, boxes))

        # 4) 中心缩放
        scale = rng.uniform(1.08, 1.20)
        zoomed = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        zh, zw = zoomed.shape[:2]
        x0, y0 = (zw - width) // 2, (zh - height) // 2
        variants.append((zoomed[y0:y0 + height, x0:x0 + width], zoom_boxes(boxes, scale)))

        for index, (aug_image, aug_boxes) in enumerate(variants[:VARIANTS], start=1):
            stem = f"{image_path.stem}_aug{index}"
            cv2.imwrite(str(image_dir / f"{stem}.jpg"), aug_image)
            write_labels(label_dir / f"{stem}.txt", aug_boxes)
            generated += 1

    print(f"Augmentation done: {len(images)} base images -> {generated} new samples "
          f"({len(images) + generated} training images total)")


if __name__ == "__main__":
    main()
