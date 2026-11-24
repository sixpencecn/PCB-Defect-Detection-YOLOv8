"""工业 PCB 图像预处理流程。

ROI 有效区域裁剪 -> 高斯滤波降噪 -> 亮度/对比度均衡 -> 尺寸归一化。
每一步均可独立调用，便于论文中说明各环节的贡献。
"""
import cv2
import numpy as np


class PCBPreprocessor:
    def __init__(self, target_size=320):
        self.target_size = target_size

    def crop_roi(self, image):
        """基于最大前景轮廓裁剪有效 PCB 区域，保留 3% 边距防止切掉边缘缺陷。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, "ROI: 未找到有效轮廓，保留原图"
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        height, width = image.shape[:2]
        # DeepPCB 图像本身已是裁剪好的板区，前景为线路而非整板轮廓。
        # 仅当轮廓在宽、高两个方向均覆盖大部分画面时才执行裁剪，避免误切缺陷。
        if w < 0.70 * width or h < 0.70 * height:
            return image, "ROI: 图像已为有效板区，保留原图"
        margin_x, margin_y = int(0.03 * width), int(0.03 * height)
        x0, y0 = max(0, x - margin_x), max(0, y - margin_y)
        x1, y1 = min(width, x + w + margin_x), min(height, y + h + margin_y)
        return image[y0:y1, x0:x1], f"ROI: 裁剪至 {x1 - x0}×{y1 - y0}"

    def denoise(self, image):
        """高斯滤波抑制细小噪点。"""
        return cv2.GaussianBlur(image, (3, 3), 0), "降噪: Gaussian 3×3"

    def balance_brightness(self, image):
        """LAB 空间 CLAHE 局部对比度增强，改善光照不均。"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(l_channel)
        result = cv2.cvtColor(cv2.merge([enhanced, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
        return result, "均衡: CLAHE clip=1.5"

    def normalize(self, image):
        """等比缩放并补边到统一尺寸（letterbox）。"""
        height, width = image.shape[:2]
        scale = min(self.target_size / width, self.target_size / height)
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.target_size, self.target_size, 3), 114, dtype=np.uint8)
        x_offset = (self.target_size - new_w) // 2
        y_offset = (self.target_size - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas, f"归一化: {self.target_size}×{self.target_size}"

    def process(self, image):
        """完整预处理流程，返回 (处理后图像, 步骤说明列表)。"""
        steps = []
        result, note = self.crop_roi(image)
        steps.append(note)
        result, note = self.denoise(result)
        steps.append(note)
        result, note = self.balance_brightness(result)
        steps.append(note)
        return result, steps
