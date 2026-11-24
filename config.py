"""PCB 缺陷检测系统全局配置。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "DeepPCB_raw" / "PCBData"
DATASET = ROOT / "data" / "yolo_dataset"
MODEL_DIR = ROOT / "models" / "pcb_yolov8n"
MODEL_PATH = MODEL_DIR / "weights" / "best.pt"
PRETRAINED_PATH = ROOT / "models" / "pretrained" / "yolov8n.pt"
OUTPUT_DIR = ROOT / "outputs"
DETECTION_DIR = OUTPUT_DIR / "detections"
REPORT_DIR = OUTPUT_DIR / "reports"
HISTORY_PATH = OUTPUT_DIR / "history.json"
METRICS_PATH = OUTPUT_DIR / "training_summary.json"

# 类别顺序与需求一致：开路、短路、针孔、毛刺、鼠咬、伪铜
CLASS_NAMES = ["open", "short", "pinhole", "burr", "mousebite", "spurious_copper"]
CLASS_NAMES_CN = ["开路", "短路", "针孔", "毛刺", "鼠咬", "伪铜"]
CLASS_NAMES_EN_LABEL = [
    "Open", "Short", "Pinhole", "Burr", "Mousebite", "Spurious Copper"
]

# DeepPCB 原始类别 ID -> 本项目类别 ID
# 原始: 1-open 2-short 3-mousebite 4-spur 5-copper 6-pin-hole
RAW_CLASS_ID_MAP = {1: 0, 2: 1, 6: 2, 4: 3, 3: 4, 5: 5}

# OpenCV BGR 颜色，分别用于六类缺陷的标注框
CLASS_COLORS_BGR = [
    (0, 0, 255),      # 开路 - 红
    (0, 165, 255),    # 短路 - 橙
    (255, 0, 0),      # 针孔 - 蓝
    (0, 255, 255),    # 毛刺 - 黄
    (255, 0, 255),    # 鼠咬 - 品红
    (60, 220, 60),    # 伪铜 - 绿
]

DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
