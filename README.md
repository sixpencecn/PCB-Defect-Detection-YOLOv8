# PCB 电路板缺陷检测系统（YOLOv8 + PyQt5）

基于 **YOLOv8** 深度学习目标检测算法与 **DeepPCB** 开源数据集构建的自动化 PCB 缺陷检测系统，可精准识别六类工业标准缺陷，替代传统人工质检方案。全流程在 **CPU** 环境下完成，代码、数据集与权重完全可复现，适合作为计算机专业毕业设计与机器视觉实训案例。

## 核心指标

| 指标 | 数值 |
|---|---|
| 精确率 Precision | 0.893 |
| 召回率 Recall | 0.783 |
| mAP50 | 0.876 |
| mAP50-95 | 0.602 |
| 单张推理耗时（CPU） | 约 0.1–0.3 秒 |

## 六类检测缺陷

| 类别 | 中文名称 | 原始数据集 ID |
|---|---|---|
| open | 开路 | 1 |
| short | 短路 | 2 |
| pinhole | 针孔 | 6 |
| burr | 毛刺 | 4 |
| mousebite | 鼠咬 | 3 |
| spurious_copper | 伪铜 | 5 |

## 功能模块

1. **图像采集**：支持本地图片批量导入，适配工业相机成像规范的 PCB 灰度/彩色图像。
2. **图像预处理**：ROI 有效区域裁剪 → 高斯滤波降噪 → CLAHE 亮度对比度均衡 → 尺寸归一化。
3. **数据集构建**：基于 DeepPCB 官方开源数据集，每类缺陷选取 50 张，提供 LabelImg 标注互转、随机翻转/亮度扰动/加噪/缩放增强脚本与自动划分。
4. **模型训练调优**：YOLOv8n 迁移训练，可调整学习率、批次、迭代次数；输出精准率、召回率、mAP 曲线与置信度/IOU 阈值对比数据。
5. **智能推理检测**：自动识别缺陷、绘制彩色检测框、输出类别与置信度、判定良品/不良品。
6. **统计与报表**：统计良品/不良品数量、六类缺陷频次、单张耗时，自动保存缺陷截图，一键导出 Excel 质检报表。
7. **PyQt5 可视化界面**：图片上传、批量检测、实时结果展示、历史记录查询、报表导出与缺陷图查看。

## 数据集说明

- 来源：[tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)，完全开源，无工业保密内容。
- 规模：1500 张 640×640 图像，10113 个标注框。
- 本项目构建：每类选取 50 张取并集得 300 张训练图，增强脚本可扩充至 1500 张；验证集 60 张、测试集 60 张均来自官方独立划分。
- 注意：官方 `trainval.txt` 中图片名为 `<id>.jpg`，实际文件为 `<id>_test.jpg`，`scripts/prepare_dataset.py` 已做后缀兼容。

## 快速开始

### 环境要求

- Python 3.8+
- 依赖见 `requirements.txt`（PyTorch、Ultralytics、OpenCV、PyQt5、openpyxl 等）

```bash
pip install -r requirements.txt
```

### 启动系统

```bash
# Windows
启动系统.bat

# Linux / macOS
bash 启动系统.sh

# 或直接运行
python main.py
```

### 重新训练（可选）

```bash
python scripts/train.py --epochs 30 --batch 16 --imgsz 320 --threads 8
```

## 项目结构

```
.
├── main.py                  # PyQt5 可视化界面入口
├── config.py                # 全局配置（类别映射、路径、超参数）
├── core/
│   ├── detector.py          # YOLOv8 推理与检测框绘制
│   ├── preprocessor.py      # 四步预处理流水线
│   ├── statistics.py        # 质检统计与 Excel 报表
│   └── image_io.py          # 中文路径图像读写兼容
├── scripts/
│   ├── prepare_dataset.py   # DeepPCB 标注转 YOLO 格式并划分数据集
│   ├── augment_dataset.py   # 离线数据增强
│   ├── train.py             # YOLOv8 训练脚本
│   ├── evaluate_thresholds.py # 置信度/IOU 阈值对比实验
│   └── labelimg_bridge.py   # LabelImg 标注互转
├── data/
│   ├── DeepPCB_raw/         # 官方开源数据集（原始文件）
│   ├── yolo_dataset/        # 转换后的 YOLO 格式数据集
│   └── sample_images/       # 六类缺陷演示样例
├── models/pcb_yolov8n/weights/best.pt  # 训练好的模型权重
├── outputs/                 # 检测结果、历史记录、报表、训练指标
├── presentation/            # 答辩演讲 PPT 与素材
├── 物料/                    # 公众号文章、答辩问答手册、介绍视频
└── requirements.txt
```

## 演示素材

- 演示图片：`data/sample_images/`
- 检测结果图：`outputs/detections/`
- 质检报表：`outputs/reports/演示质检报表.xlsx`
- 训练指标：`outputs/training_summary.json`
- 阈值对比：`outputs/threshold_comparison.csv`
- 答辩 PPT：`presentation/PCB缺陷检测系统全面演讲/`
- 答辩指南与问答手册：`物料/答辩演讲指南与问答手册.md`
- 系统介绍视频：`物料/PCB缺陷检测系统_介绍视频.mp4`

## 阈值实验结论

| 置信度 | 精确率 | 召回率 | 适用场景 |
|---|---|---|---|
| 0.25 | 0.864 | 0.817 | 低漏检优先 |
| 0.50 | 0.927 | 0.744 | 均衡 |
| 0.65 | 0.959 | 0.589 | 低误报优先 |

IOU 阈值在 0.35–0.6 范围内对结果影响很小，因为 PCB 缺陷目标尺度小、位置相对固定。

## 局限与后续工作

- 320 输入尺寸下极小缺陷（针孔、开路）召回率仍有提升空间，可提升至 640 或增加小目标检测层。
- 目前支持离线图片检测，后续可接入工业相机实时视频流。
- 可补充预处理消融实验与更多超参数对比。

## License

数据集来自 [DeepPCB](https://github.com/tangsanli5201/DeepPCB) 开源仓库，本项目代码用于毕业设计与学习交流。
