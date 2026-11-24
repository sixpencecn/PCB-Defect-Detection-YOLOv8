"""YOLOv8 推理封装：缺陷识别、标注框绘制、良品/不良品判定。"""
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (CLASS_COLORS_BGR, CLASS_NAMES_CN, DEFAULT_CONF, DEFAULT_IOU,
                    MODEL_PATH)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def load_chinese_font(size):
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


class PCBDetector:
    def __init__(self, weights_path=MODEL_PATH, conf=DEFAULT_CONF, iou=DEFAULT_IOU, imgsz=320):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"模型权重不存在: {weights_path}")
        from ultralytics import YOLO
        self.model = YOLO(str(weights_path))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.font = load_chinese_font(15)

    def predict(self, image_bgr, conf=None, iou=None):
        """返回 (缺陷列表, 耗时毫秒)。缺陷字典含类别 ID/名称、置信度、坐标。"""
        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou
        start = time.perf_counter()
        result = self.model.predict(
            source=image_bgr,
            imgsz=self.imgsz,
            conf=conf,
            iou=iou,
            device="cpu",
            verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for detection in result.boxes:
                x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().tolist()
                class_id = int(detection.cls[0].cpu().item())
                score = float(detection.conf[0].cpu().item())
                boxes.append({
                    "class_id": class_id,
                    "class_name": CLASS_NAMES_CN[class_id],
                    "confidence": score,
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                })
        boxes.sort(key=lambda item: -item["confidence"])
        return boxes, elapsed_ms

    def annotate(self, image_bgr, boxes):
        """在图像上绘制检测框与中文类别标签。"""
        pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        for box in boxes:
            color = tuple(reversed(CLASS_COLORS_BGR[box["class_id"]]))
            x1, y1 = max(0, int(box["x1"])), max(0, int(box["y1"]))
            x2, y2 = min(pil_image.width - 1, int(box["x2"])), min(pil_image.height - 1, int(box["y2"]))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f"{box['class_name']} {box['confidence']:.2f}"
            text_box = draw.textbbox((0, 0), label, font=self.font)
            text_w, text_h = text_box[2] - text_box[0], text_box[3] - text_box[1]
            label_y = y1 - text_h - 6 if y1 - text_h - 6 > 0 else y1 + 2
            draw.rectangle([x1, label_y, x1 + text_w + 8, label_y + text_h + 5], fill=color)
            draw.text((x1 + 4, label_y + 1), label, font=self.font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
