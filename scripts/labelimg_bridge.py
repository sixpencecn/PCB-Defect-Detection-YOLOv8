"""LabelImg 标注校验与微调桥梁。

export: YOLO txt -> Pascal VOC XML，可在 LabelImg 中打开校验/微调。
import: LabelImg 保存的 XML -> YOLO txt，回写训练集标注。

用法：
    python scripts/labelimg_bridge.py export --split train
    python scripts/labelimg_bridge.py import --split train
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CLASS_NAMES, DATASET

XML_DIR = ROOT / "tools" / "labelimg_xml"


def export_xml(split):
    XML_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted((DATASET / "images" / split).glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = DATASET / "labels" / split / f"{image_path.stem}.txt"
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "filename").text = image_path.name
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(annotation, "path").text = str(image_path)
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                if not 0 <= class_id < len(CLASS_NAMES):
                    continue
                x1 = max(0, (cx - bw / 2) * width)
                y1 = max(0, (cy - bh / 2) * height)
                x2 = min(width, (cx + bw / 2) * width)
                y2 = min(height, (cy + bh / 2) * height)
                obj = ET.SubElement(annotation, "object")
                ET.SubElement(obj, "name").text = CLASS_NAMES[class_id]
                ET.SubElement(obj, "difficult").text = "0"
                box = ET.SubElement(obj, "bndbox")
                for tag, value in (("xmin", x1), ("ymin", y1), ("xmax", x2), ("ymax", y2)):
                    ET.SubElement(box, tag).text = str(int(round(value)))
        ET.ElementTree(annotation).write(
            str(XML_DIR / f"{image_path.stem}.xml"), encoding="utf-8", xml_declaration=True
        )
        count += 1
    print(f"Exported {count} LabelImg XML files to {XML_DIR}")


def import_xml(split):
    name_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    count = 0
    for xml_path in sorted(XML_DIR.glob("*.xml")):
        image_path = DATASET / "images" / split / f"{xml_path.stem}.jpg"
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        root = ET.parse(str(xml_path)).getroot()
        lines = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name not in name_to_id:
                continue
            box = obj.find("bndbox")
            x1 = float(box.findtext("xmin"))
            y1 = float(box.findtext("ymin"))
            x2 = float(box.findtext("xmax"))
            y2 = float(box.findtext("ymax"))
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height
            lines.append(f"{name_to_id[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (DATASET / "labels" / split / f"{xml_path.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        count += 1
    print(f"Imported {count} LabelImg XML files back into {split} labels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("direction", choices=["export", "import"])
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()
    if args.direction == "export":
        export_xml(args.split)
    else:
        import_xml(args.split)
