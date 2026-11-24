"""兼容中文路径的 OpenCV 图像读写封装。"""
from pathlib import Path

import cv2
import numpy as np


def imread(path):
    """读取图像，支持中文/空格路径。失败返回 None。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def imwrite(path, image):
    """保存图像，支持中文/空格路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix if path.suffix else ".jpg"
    success, buffer = cv2.imencode(extension, image)
    if success:
        buffer.tofile(str(path))
    return success
