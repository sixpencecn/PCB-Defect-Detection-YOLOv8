"""DeepPCB 原始标注 -> YOLOv8 数据集。

按需求从官方 trainval 划分中每类选取 50 张图片作为训练集，
额外每类选取 10 张作为验证集；测试集取自官方 test 划分（每类 10 张）。
一张图片可能包含多个类别的缺陷，因此采用"按类选取后取并集"的策略。
"""
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CLASS_NAMES, DATASET, RAW_CLASS_ID_MAP, RAW_DATA

PER_CLASS_TRAIN = 50
PER_CLASS_VAL = 10
PER_CLASS_TEST = 10
SEED = 2026


def parse_split(list_name):
    """解析 one line: image_path annotation_path"""
    items = []
    for line in (RAW_DATA / list_name).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        img_rel, ann_rel = parts[0], parts[1]
        ann_path = RAW_DATA / ann_rel
        if not ann_path.exists():
            continue
        # 官方清单写的是 <id>.jpg，仓库实际文件名为 <id>_test.jpg
        image_path = RAW_DATA / img_rel
        if not image_path.exists():
            image_path = image_path.with_name(f"{image_path.stem}_test.jpg")
        if not image_path.exists():
            continue
        boxes = []
        for raw_line in ann_path.read_text(errors="ignore").splitlines():
            values = raw_line.replace(",", " ").split()
            if len(values) < 5:
                continue
            try:
                x1, y1, x2, y2, type_id = map(float, values[:5])
                type_id = int(type_id)
            except ValueError:
                continue
            if type_id in RAW_CLASS_ID_MAP and x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2, RAW_CLASS_ID_MAP[type_id]))
        if not boxes:
            continue
        items.append({
            "rel": img_rel,
            "image": image_path,
            "boxes": boxes,
            "classes": sorted({b[4] for b in boxes}),
        })
    return items


def select_images(items, per_class, seed, exclude=frozenset()):
    """每类选取 per_class 张图片，返回并集（保持确定性）。"""
    rng = random.Random(seed)
    pool = defaultdict(list)
    for item in items:
        if item["rel"] in exclude:
            continue
        for class_id in item["classes"]:
            pool[class_id].append(item["rel"])
    chosen = set()
    for class_id in range(len(CLASS_NAMES)):
        candidates = [r for r in pool[class_id] if r not in chosen]
        rng.shuffle(candidates)
        chosen.update(candidates[:per_class])
    return chosen


def to_yolo_lines(boxes, width, height):
    lines = []
    for x1, y1, x2, y2, class_id in boxes:
        x1 = min(max(0.0, x1), float(width - 1))
        x2 = min(max(0.0, x2), float(width - 1))
        y1 = min(max(0.0, y1), float(height - 1))
        y2 = min(max(0.0, y2), float(height - 1))
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        if bw <= 0 or bh <= 0:
            continue
        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main():
    trainval = parse_split("trainval.txt")
    test = parse_split("test.txt")
    by_rel = {item["rel"]: item for item in trainval + test}

    train_ids = select_images(trainval, PER_CLASS_TRAIN, SEED)
    val_ids = select_images(trainval, PER_CLASS_VAL, SEED + 1, exclude=train_ids)
    test_ids = select_images(test, PER_CLASS_TEST, SEED + 2)

    for split in ("train", "val", "test"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {}
    for split, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        box_counter = Counter()
        image_counter = Counter()
        for rel in sorted(ids):
            item = by_rel[rel]
            image = cv2.imread(str(item["image"]))
            if image is None:
                print(f"[WARN] skip unreadable image: {item['image']}")
                continue
            height, width = image.shape[:2]
            stem = Path(rel).stem
            shutil.copyfile(str(item["image"]), str(DATASET / "images" / split / f"{stem}.jpg"))
            lines = to_yolo_lines(item["boxes"], width, height)
            (DATASET / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            for class_id in {int(line.split()[0]) for line in lines}:
                image_counter[class_id] += 1
            box_counter.update(int(line.split()[0]) for line in lines)
        stats[split] = {
            "images": len(list((DATASET / "images" / split).glob("*.jpg"))),
            "images_per_class": {CLASS_NAMES[i]: image_counter[i] for i in range(len(CLASS_NAMES))},
            "boxes_per_class": {CLASS_NAMES[i]: box_counter[i] for i in range(len(CLASS_NAMES))},
        }

    yaml_text = (
        f"path: {DATASET.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names: [" + ", ".join(CLASS_NAMES) + "]\n"
    )
    (DATASET / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (DATASET / "dataset_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Dataset prepared:", DATASET)
    for split, info in stats.items():
        print(f"  {split:5s}: {info['images']} images, per-class images = {info['images_per_class']}")
    print("  boxes:", {s: sum(i["boxes_per_class"].values()) for s, i in stats.items()})


if __name__ == "__main__":
    main()
