# -*- coding: utf-8 -*-
"""生成 PCB 缺陷检测系统介绍视频（本地素材版，无需额外模型/积分）。

画面使用项目真实检测图与训练曲线，配中文解说字幕。
输出：物料/PCB缺陷检测系统_介绍视频.mp4
"""
from pathlib import Path
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent.parent
ASSETS = BASE / "presentation" / "PCB缺陷检测系统全面演讲" / "assets"
OUT = BASE / "物料" / "PCB缺陷检测系统_介绍视频.mp4"

W, H = 1280, 720
FPS = 24
FONT_REG = "C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"

NAVY = (14, 35, 64)
BLUE = (30, 111, 219)
LIGHT = (244, 248, 253)
INK = (20, 42, 70)
GREY = (95, 116, 142)
WHITE = (255, 255, 255)
ACCENT = (0, 176, 212)


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def new_bg(color=WHITE):
    im = Image.new("RGB", (W, H), color)
    return im


def draw_text(im, xy, text, size=28, color=INK, bold=False, anchor="la", spacing=8):
    d = ImageDraw.Draw(im)
    d.text(xy, text, font=font(size, bold), fill=color, anchor=anchor, spacing=spacing)
    return im


_GRAD_CACHE = None


def gradient_dark(im):
    global _GRAD_CACHE
    if _GRAD_CACHE is None:
        yy = np.arange(H, dtype=np.float32)[:, None] / H
        xx = np.arange(W, dtype=np.float32)[None, :] / W
        glow = np.sin(xx * math.pi)
        r = 10 + 12 * yy + 8 * glow
        g = 28 + 22 * yy + 14 * glow
        b = 56 + 34 * yy + 20 * glow
        arr = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
        _GRAD_CACHE = Image.fromarray(arr)
    im.paste(_GRAD_CACHE)
    return im


def rounded_panel(im, box, fill=(255, 255, 255), outline=None, radius=18, width=2):
    d = ImageDraw.Draw(im)
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    return im


def fit_image(src, box, zoom=1.0, pan=(0.5, 0.5)):
    """按 box 等比裁切并缩放，zoom>1 产生缓慢推进效果。"""
    bw, bh = box
    iw, ih = src.size
    scale = max(bw / iw, bh / ih) * zoom
    nw, nh = iw * scale, ih * scale
    left = (nw - bw) * pan[0]
    top = (nh - bh) * pan[1]
    r = src.resize((int(nw), int(nh)), Image.Resampling.LANCZOS)
    return r.crop((int(left), int(top), int(left) + bw, int(top) + bh))


def paste_rounded(im, patch, xy, radius=12):
    mask = Image.new("L", patch.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, patch.size[0], patch.size[1]), radius=radius, fill=255)
    im.paste(patch, xy, mask)


def load_asset(name):
    p = ASSETS / name
    if not p.exists():
        p = BASE / "物料" / "images" / name
    return Image.open(p).convert("RGB")


# ---------- 场景 ----------

def scene_cover(t):
    im = gradient_dark(new_bg())
    d = ImageDraw.Draw(im)
    # 装饰线
    for i in range(7):
        y = 120 + i * 78
        d.line([(90, y), (W - 90, y)], fill=(38, 72, 116), width=1)
    pulse = int(120 + 60 * math.sin(t * math.pi))
    d.rounded_rectangle((92, 156, 100, 232), radius=4, fill=(pulse, 190, 240))
    draw_text(im, (130, 150), "毕业设计 · 工业视觉检测", 26, (150, 195, 240), bold=True)
    draw_text(im, (130, 232), "PCB 电路板", 78, WHITE, bold=True)
    draw_text(im, (130, 330), "缺陷检测系统", 78, WHITE, bold=True)
    draw_text(im, (134, 458), "基于 YOLOv8 深度学习算法的自动化质检方案", 32, (190, 216, 244))
    tags = ["YOLOv8n", "DeepPCB 开源数据集", "PyQt5 桌面系统", "CPU 可复现"]
    x = 134
    for tag in tags:
        w = int(d.textlength(tag, font=font(20))) + 32
        d.rounded_rectangle((x, 530, x + w, 578), radius=24, fill=(28, 62, 104), outline=(70, 120, 175), width=1)
        draw_text(im, (x + 16, 542), tag, 20, (210, 231, 252))
        x += w + 14
    draw_text(im, (134, 640), "六类典型缺陷  |  批量检测  |  质检报表", 22, (140, 180, 220))
    return im


def scene_pain(t):
    im = new_bg(LIGHT)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 118), fill=NAVY)
    draw_text(im, (80, 34), "01  为什么要做", 40, WHITE, bold=True)
    draw_text(im, (80, 154), "人工目检的三个痛点", 44, INK, bold=True)
    d.line([(80, 216), (230, 216)], fill=BLUE, width=5)
    cards = [
        ("容易漏检", "缺陷尺度小、纹理复杂\n长时间作业后注意力下降"),
        ("尺度不一", "不同班次判定标准不同\n结果依赖个人经验"),
        ("不可追溯", "检测记录靠人工书写\n问题回溯缺少量化依据"),
    ]
    for i, (title, body) in enumerate(cards):
        appear = min(1.0, max(0.0, (t - 0.08 - i * 0.16) * 4))
        if appear <= 0:
            continue
        x = 80 + i * 385
        y = 268 + int((1 - appear) * 26)
        rounded_panel(im, (x, y, x + 350, y + 300), fill=WHITE, outline=(214, 226, 240), radius=18)
        d.rectangle((x, y, x + 8, y + 300), fill=BLUE if i == 0 else ACCENT)
        draw_text(im, (x + 32, y + 32), f"0{i+1}", 26, (160, 180, 205), bold=True)
        draw_text(im, (x + 32, y + 86), title, 34, INK, bold=True)
        draw_text(im, (x + 32, y + 158), body, 22, GREY, spacing=10)
    if t > 0.62:
        rounded_panel(im, (80, 606, W - 80, 676), fill=(226, 239, 253), radius=14)
        draw_text(im, (104, 622), "解决思路：把「看板判断」变成可复现的图像推理流程", 26, BLUE, bold=True)
    return im


def scene_six(t):
    im = new_bg(LIGHT)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 118), fill=NAVY)
    draw_text(im, (80, 34), "02  数据集与检测目标", 40, WHITE, bold=True)
    draw_text(im, (80, 148), "六类工业标准缺陷", 44, INK, bold=True)
    draw_text(im, (80, 206), "DeepPCB 开源数据集  ·  1500 张图像  ·  10113 个标注框", 24, GREY)
    names = ["open  开路", "short  短路", "pinhole  针孔", "burr  毛刺", "mousebite  鼠咬", "spurious copper  伪铜"]
    files = ["detect_open.jpg", "detect_short.jpg", "detect_pinhole.jpg", "detect_burr.jpg", "detect_mousebite.jpg", "detect_copper.jpg"]
    positions = [(80, 258), (475, 258), (870, 258), (80, 500), (475, 500), (870, 500)]
    for i, (name, fn, pos) in enumerate(zip(names, files, positions)):
        path = ASSETS / fn
        if not path.exists():
            continue
        src = load_asset(fn)
        zoom = 1.0 + 0.06 * t
        patch = fit_image(src, (330, 180), zoom=zoom, pan=(0.5, 0.48))
        rounded_panel(im, (pos[0] - 4, pos[1] - 4, pos[0] + 334, pos[1] + 208), fill=WHITE, outline=(208, 222, 238), radius=12)
        paste_rounded(im, patch, pos, radius=10)
        d.rounded_rectangle((pos[0] + 8, pos[1] + 186, pos[0] + 250, pos[1] + 220), radius=8, fill=(20, 42, 70))
        draw_text(im, (pos[0] + 18, pos[1] + 192), name, 17, WHITE, bold=True)
    return im


def scene_pipeline(t):
    im = new_bg(LIGHT)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 118), fill=NAVY)
    draw_text(im, (80, 34), "03  技术路线", 40, WHITE, bold=True)
    draw_text(im, (80, 154), "从图像到报表的完整闭环", 44, INK, bold=True)
    steps = [
        ("图像采集", "本地图片批量导入"),
        ("预处理", "ROI·降噪·均衡·归一化"),
        ("数据集构建", "每类50张·增强·划分"),
        ("模型训练", "YOLOv8n 迁移训练"),
        ("推理与统计", "检测框·OK/NG·Excel"),
    ]
    n = len(steps)
    gap = 18
    bw = (W - 160 - gap * (n - 1)) // n
    for i, (title, sub) in enumerate(steps):
        x = 80 + i * (bw + gap)
        appear = min(1.0, max(0.0, (t - i * 0.08) * 4))
        y = 300 + int((1 - appear) * 30)
        rounded_panel(im, (x, y, x + bw, y + 220), fill=WHITE, outline=(210, 224, 240), radius=16)
        d.rounded_rectangle((x + 18, y + 22, x + 54, y + 66), radius=10, fill=BLUE)
        draw_text(im, (x + 36, y + 44), str(i + 1), 22, WHITE, bold=True, anchor="mm")
        draw_text(im, (x + 18, y + 96), title, 24, INK, bold=True)
        draw_text(im, (x + 18, y + 146), sub, 16, GREY, spacing=6)
        if i < n - 1:
            draw_text(im, (x + bw + 2, y + 96), "→", 24, BLUE, bold=True, anchor="mm")
    rounded_panel(im, (80, 580, W - 80, 660), fill=(226, 239, 253), radius=14)
    draw_text(im, (104, 600), "全流程在本地 CPU 完成  ·  代码 / 数据集 / 权重均可复现", 26, BLUE, bold=True)
    return im


def scene_results(t):
    im = new_bg(LIGHT)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 118), fill=NAVY)
    draw_text(im, (80, 34), "04  实验结果", 40, WHITE, bold=True)
    # 左侧曲线
    src = load_asset("curve_results.png")
    zoom = 1.0 + 0.05 * t
    patch = fit_image(src, (700, 430), zoom=zoom)
    rounded_panel(im, (66, 146, 784, 596), fill=WHITE, outline=(210, 224, 240), radius=14)
    paste_rounded(im, patch, (70, 150), radius=10)
    # 右侧指标卡
    metrics = [("精确率 Precision", "0.893"), ("召回率 Recall", "0.783"), ("mAP50", "0.876"), ("mAP50-95", "0.602")]
    for i, (label, val) in enumerate(metrics):
        y = 150 + i * 112
        rounded_panel(im, (812, y, W - 70, y + 96), fill=WHITE, outline=(212, 226, 242), radius=14)
        draw_text(im, (836, y + 14), label, 19, GREY)
        draw_text(im, (836, y + 44), val, 38, BLUE, bold=True)
    rounded_panel(im, (70, 620, W - 70, 690), fill=(226, 239, 253), radius=12)
    draw_text(im, (92, 638), "阈值权衡：conf 0.25 → 精确率 0.864 / 召回率 0.817    |    0.65 → 0.959 / 0.589", 21, BLUE, bold=True)
    return im


def scene_system(t):
    im = new_bg(LIGHT)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 118), fill=NAVY)
    draw_text(im, (80, 34), "05  系统实现", 40, WHITE, bold=True)
    draw_text(im, (80, 150), "PyQt5 桌面端：检测 · 统计 · 追溯一体化", 42, INK, bold=True)
    src = load_asset("detect_multi.jpg")
    patch = fit_image(src, (620, 350), zoom=1.02 + 0.05 * t)
    rounded_panel(im, (70, 232, 710, 602), fill=WHITE, outline=(210, 224, 240), radius=14)
    paste_rounded(im, patch, (74, 236), radius=10)
    feats = [
        ("批量检测", "多选图片，后台线程推理，界面不卡顿"),
        ("结果可视化", "彩色检测框 + 中文类别 + 置信度"),
        ("历史追溯", "记录判定、耗时与结果图路径"),
        ("Excel 报表", "明细 / 缺陷统计 / 质检汇总三表"),
    ]
    for i, (title, desc) in enumerate(feats):
        y = 240 + i * 92
        rounded_panel(im, (742, y, W - 70, y + 78), fill=WHITE, outline=(212, 226, 242), radius=12)
        d.rounded_rectangle((742, y, 750, y + 78), radius=4, fill=ACCENT if i % 2 else BLUE)
        draw_text(im, (768, y + 12), title, 24, INK, bold=True)
        draw_text(im, (768, y + 46), desc, 16, GREY)
    return im


def scene_end(t):
    im = gradient_dark(new_bg())
    d = ImageDraw.Draw(im)
    draw_text(im, (W // 2, 268), "感谢各位老师", 72, WHITE, bold=True, anchor="mm")
    d.line([(W // 2 - 180, 330), (W // 2 + 180, 330)], fill=(70, 150, 220), width=4)
    draw_text(im, (W // 2, 398), "敬请批评指正", 38, (180, 214, 246), anchor="mm")
    draw_text(im, (W // 2, 500), "github.com/sixpencecn/PCB-Defect-Detection-YOLOv8", 22, (130, 180, 225), anchor="mm")
    return im


SCENES = [
    (scene_cover, 5.0),
    (scene_pain, 7.0),
    (scene_six, 7.0),
    (scene_pipeline, 6.5),
    (scene_results, 7.5),
    (scene_system, 7.0),
    (scene_end, 5.0),
]
FADE = 0.55


def main():
    frames = []
    for fn, dur in SCENES:
        total = int(dur * FPS)
        for i in range(total):
            frames.append(np.array(fn(i / total))[:, :, ::-1])
    # 交叉淡化
    out_frames = []
    offset = 0
    for idx, (fn, dur) in enumerate(SCENES):
        n = int(dur * FPS)
        start = offset
        end = start + n
        seg = frames[start:end]
        if idx == 0:
            out_frames.extend(seg[:-int(FADE * FPS)])
        else:
            k = int(FADE * FPS)
            prev_tail = frames[max(0, start - k):start]
            if len(prev_tail) < k:
                prev_tail = [frames[start - 1]] * (k - len(prev_tail)) + prev_tail
            for j in range(k):
                a = prev_tail[j].astype(np.float32)
                b = seg[j].astype(np.float32)
                w = (j + 1) / (k + 1)
                out_frames.append((a * (1 - w) + b * w).astype(np.uint8))
            out_frames.extend(seg[k:])
        offset = end
    last_k = int(FADE * FPS)
    out_frames = out_frames[:-last_k] + frames[-last_k:]
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter 无法打开")
    for f in out_frames:
        writer.write(f)
    writer.release()
    print(f"视频生成完成: {OUT}")
    print(f"时长: {len(out_frames) / FPS:.1f} 秒 | 分辨率: {W}x{H} | 帧率: {FPS}")


if __name__ == "__main__":
    main()
