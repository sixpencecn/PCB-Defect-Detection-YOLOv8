"""PCB 电路板缺陷检测系统 —— PyQt5 可视化界面入口。"""
import os
import sys
from datetime import datetime
from pathlib import Path


def _configure_qt_plugins():
    """绑定当前解释器配套的 Qt 平台插件，避免外部环境变量污染导致启动失败。

    PyCharm、Anaconda 等工具会注入 QT_QPA_PLATFORM_PLUGIN_PATH，
    指向它们自带的（可能不兼容的）qwindows.dll，造成
    "no Qt platform plugin could be initialized" 错误。
    """
    try:
        import PyQt5
        root = Path(PyQt5.__file__).resolve().parent
        for candidate in (root / "Qt5" / "plugins" / "platforms",
                          root / "Qt" / "plugins" / "platforms"):
            if (candidate / "qwindows.dll").exists():
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(candidate)
                break
    except Exception:
        pass
    # 清除可能指向错误平台的外部设置（如 offscreen / minimal）
    for key in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM"):
        os.environ.pop(key, None)


_configure_qt_plugins()

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QDoubleSpinBox,
                             QFileDialog, QFrame, QGroupBox, QHBoxLayout, QHeaderView,
                             QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                             QStatusBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from config import (CLASS_NAMES_CN, DEFAULT_CONF, DEFAULT_IOU, DETECTION_DIR,
                    HISTORY_PATH, MODEL_PATH, REPORT_DIR)
from core.detector import PCBDetector
from core.image_io import imread, imwrite
from core.preprocessor import PCBPreprocessor
from core.statistics import DetectionRecord, QualityStatistics

IMAGE_FILTER = "图片文件 (*.jpg *.jpeg *.png *.bmp)"


class DetectionWorker(QThread):
    """批量检测后台线程，避免 UI 冻结。"""
    progress = pyqtSignal(int, int, object)
    finished_all = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, paths, use_preprocess, conf, iou, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.use_preprocess = use_preprocess
        self.conf = conf
        self.iou = iou

    def run(self):
        try:
            detector = PCBDetector(conf=self.conf, iou=self.iou)
        except Exception as exc:
            self.failed.emit(f"模型加载失败：{exc}")
            return
        # 首帧包含 CPU 算子初始化开销，提前预热，避免计入单张检测耗时
        detector.predict(np.zeros((320, 320, 3), dtype=np.uint8))
        preprocessor = PCBPreprocessor()
        records = []
        for index, path in enumerate(self.paths, start=1):
            try:
                image = imread(path)
                if image is None:
                    raise ValueError("图片无法读取")
                if self.use_preprocess:
                    image, _ = preprocessor.process(image)
                boxes, elapsed_ms = detector.predict(image, self.conf, self.iou)
                annotated = detector.annotate(image, boxes)

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                output_path = DETECTION_DIR / f"{stamp}_{Path(path).stem}.jpg"
                imwrite(output_path, annotated)

                record = DetectionRecord(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    filename=Path(path).name,
                    source=str(path),
                    verdict="NG" if boxes else "OK",
                    top_confidence=boxes[0]["confidence"] if boxes else 0.0,
                    defects=boxes,
                    elapsed_ms=elapsed_ms,
                    annotated_path=str(output_path),
                )
                records.append(record)
                self.progress.emit(index, len(self.paths), record)
            except Exception as exc:
                self.failed.emit(f"{Path(path).name}: {exc}")
        self.finished_all.emit(records)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB电路板缺陷检测系统 | YOLOv8 · 工业视觉质检")
        self.resize(1240, 790)
        self.setMinimumSize(1024, 680)

        self.image_paths = []
        self.statistics = QualityStatistics()
        self.statistics.load_history(HISTORY_PATH)
        self._pixmap_cache = {}
        self.worker = None
        self._init_ui()
        self._refresh_record_table()
        self._refresh_stats_panel()
        self._update_status_model()

    # ---------------- UI ----------------
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(9)

        title = QLabel("PCB电路板缺陷检测系统 — 基于 YOLOv8")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#1a4f9c; font-size:23px; font-weight:bold; padding:4px;")
        layout.addWidget(title)

        subtitle = QLabel("DeepPCB 开源数据集训练 · 六类工业标准缺陷 · 支持批量检测与 Excel 质检报表")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color:#697386; font-size:11px;")
        layout.addWidget(subtitle)

        button_layout = QHBoxLayout()
        self.upload_btn = QPushButton("上传图片(可多选)")
        self.detect_btn = QPushButton("一键检测")
        self.detect_btn.setObjectName("primary")
        self.export_btn = QPushButton("导出Excel报表")
        self.clear_btn = QPushButton("清空历史")
        for button in (self.upload_btn, self.detect_btn, self.export_btn, self.clear_btn):
            button.setMinimumHeight(35)
            button_layout.addWidget(button)

        self.preprocess_cb = QCheckBox("启用预处理(抑制光照不均/反光)")
        self.preprocess_cb.setToolTip("ROI 裁剪 + 高斯降噪 + 亮度对比度均衡，适用于光照不均或噪点较多的图像")
        button_layout.addWidget(self.preprocess_cb)
        button_layout.addStretch(1)
        button_layout.addWidget(QLabel("置信度:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(DEFAULT_CONF)
        self.conf_spin.setFixedWidth(70)
        button_layout.addWidget(self.conf_spin)
        layout.addLayout(button_layout)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)

        image_group = QGroupBox("图像与检测结果")
        image_layout = QHBoxLayout(image_group)
        image_layout.setSpacing(8)
        self.original_label = self._create_image_label("等待上传图片")
        self.result_label = self._create_image_label("尚未检测\n点击[一键检测]开始")
        image_layout.addWidget(self._wrap_image_panel("原图", self.original_label), 1)
        image_layout.addWidget(self._wrap_image_panel("检测结果图(带框)", self.result_label), 1)
        main_layout.addWidget(image_group, 3)

        right_layout = QVBoxLayout()
        history_group = QGroupBox("检测历史")
        history_layout = QVBoxLayout(history_group)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels(
            ["序号", "时间", "来源", "判定", "置信度", "缺陷数", "耗时(ms)"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.cellClicked.connect(self._on_history_clicked)
        history_layout.addWidget(self.history_table)
        right_layout.addWidget(history_group, 3)

        stats_group = QGroupBox("质检统计")
        stats_layout = QVBoxLayout(stats_group)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color:#243447; font-size:12px; padding:2px;")
        stats_layout.addWidget(self.summary_label)
        self.defect_table = QTableWidget()
        self.defect_table.setColumnCount(2)
        self.defect_table.setHorizontalHeaderLabels(["缺陷类别", "出现次数"])
        self.defect_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.defect_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.defect_table.verticalHeader().setVisible(False)
        self.defect_table.setAlternatingRowColors(True)
        for row, name in enumerate(CLASS_NAMES_CN):
            self.defect_table.insertRow(row)
            self.defect_table.setItem(row, 0, QTableWidgetItem(name))
            self.defect_table.setItem(row, 1, QTableWidgetItem("0"))
        stats_layout.addWidget(self.defect_table)
        right_layout.addWidget(stats_group, 2)
        main_layout.addLayout(right_layout, 2)
        layout.addLayout(main_layout)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(180)
        self.progress_bar.setRange(0, 100)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("就绪 · 请上传 PCB 图片")

        self.upload_btn.clicked.connect(self._upload_images)
        self.detect_btn.clicked.connect(self._start_detection)
        self.export_btn.clicked.connect(self._export_report)
        self.clear_btn.clicked.connect(self._clear_history)
        self.menuBar().addAction("使用说明", self._show_help)

        self.setStyleSheet("""
            QMainWindow { background:#f4f7fb; }
            QGroupBox { background:#ffffff; border:1px solid #d7e0ec; border-radius:6px;
                        margin-top:9px; font-weight:bold; color:#2c5aa0; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
            QPushButton { background:#ffffff; border:1px solid #b9c6d8; border-radius:5px;
                          padding:6px 12px; color:#243447; }
            QPushButton:hover { background:#e9f2ff; }
            QPushButton#primary { background:#2f6fdb; color:#ffffff; border:1px solid #2f6fdb;
                                  font-weight:bold; }
            QPushButton#primary:hover { background:#255fbe; }
            QTableWidget { background:#ffffff; alternate-background-color:#f6f9fd;
                           gridline-color:#e3e9f1; border:1px solid #d7e0ec; border-radius:4px; }
            QHeaderView::section { background:#eef3fa; border:1px solid #d7e0ec;
                                   padding:5px; font-weight:bold; color:#33445c; }
            QStatusBar { background:#ffffff; color:#52627a; }
        """)

    def _create_image_label(self, placeholder):
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(360, 380)
        label.setStyleSheet(
            "background:#fbfcfe; border:1px dashed #c5d1e0; border-radius:4px; color:#9aa7b8;"
        )
        return label

    def _wrap_image_panel(self, title, label):
        panel = QFrame()
        panel.setStyleSheet("background:#ffffff; border:1px solid #e1e8f1; border-radius:5px;")
        panel_layout = QVBoxLayout(panel)
        caption = QLabel(title)
        caption.setStyleSheet("color:#3f5471; font-weight:bold; padding:4px 8px;")
        panel_layout.addWidget(caption)
        panel_layout.addWidget(label, 1)
        return panel

    def _update_status_model(self):
        if MODEL_PATH.exists():
            self.status_bar.showMessage(f"模型就绪 · {MODEL_PATH.name} · 等待检测", 4000)
        else:
            self.detect_btn.setEnabled(False)
            self.status_bar.showMessage("警告：未找到训练权重，请先运行 python scripts/train.py")

    # ---------------- 图片显示 ----------------
    def _set_label_image(self, label, image_bgr):
        if image_bgr is None:
            self._pixmap_cache.pop(label, None)
            label.setPixmap(QPixmap())
            return
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        self._pixmap_cache[label] = QPixmap.fromImage(qimage)
        self._refresh_image_label(label)

    def _refresh_image_label(self, label):
        pixmap = self._pixmap_cache.get(label)
        if pixmap is None or label.width() < 10 or label.height() < 10:
            return
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_image_label(self.original_label)
        self._refresh_image_label(self.result_label)

    # ---------------- 业务逻辑 ----------------
    def _upload_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择 PCB 图片", "", IMAGE_FILTER)
        if not paths:
            return
        self.image_paths = paths
        first = imread(paths[0])
        self._set_label_image(self.original_label, first)
        self._set_label_image(self.result_label, None)
        self.result_label.setText("尚未检测\n点击[一键检测]开始")
        self.status_bar.showMessage(f"已载入 {len(paths)} 张图片 · 点击[一键检测]开始")

    def _start_detection(self):
        if not self.image_paths:
            QMessageBox.information(self, "提示", "请先上传图片。")
            return
        if not MODEL_PATH.exists():
            QMessageBox.warning(self, "模型缺失", "未找到训练权重，请先运行训练脚本。")
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = DetectionWorker(
            self.image_paths,
            self.preprocess_cb.isChecked(),
            self.conf_spin.value(),
            DEFAULT_IOU,
            self,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.detect_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("正在检测，请稍候…")
        self.worker.start()

    def _append_record_row(self, record):
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        values = [
            str(row + 1), record.timestamp, record.filename, record.verdict,
            f"{record.top_confidence:.2f}", str(len(record.defects)), f"{record.elapsed_ms:.0f}",
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignCenter)
            if column == 3:
                item.setForeground(QColor("red" if value == "NG" else "green"))
                item.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
            self.history_table.setItem(row, column, item)

    def _refresh_record_table(self):
        self.history_table.setRowCount(0)
        for record in self.statistics.records:
            self._append_record_row(record)

    def _refresh_stats_panel(self):
        summary = self.statistics.summary()
        self.summary_label.setText(
            f"检测总数 <b>{summary['total']}</b> ｜ 良品 <b style='color:#008000'>{summary['ok']}</b> ｜ "
            f"不良品 <b style='color:#c00000'>{summary['ng']}</b> ｜ 不良率 <b>{summary['ng_rate']:.1f}%</b> ｜ "
            f"平均耗时 <b>{summary['avg_ms']:.0f} ms</b>"
        )
        counts = self.statistics.defect_counts()
        for row, name in enumerate(CLASS_NAMES_CN):
            self.defect_table.setItem(row, 1, QTableWidgetItem(str(counts.get(name, 0))))

    def _on_progress(self, index, total, record):
        self.statistics.add(record)
        self._append_record_row(record)
        self.history_table.scrollToBottom()
        self._refresh_stats_panel()
        self._set_label_image(self.original_label, imread(record.source))
        self._set_label_image(self.result_label, imread(record.annotated_path))
        self.progress_bar.setValue(int(index / total * 100))
        self.status_bar.showMessage(f"检测中 {index}/{total} · {record.filename} · {record.verdict}")

    def _on_finished(self, records):
        self.detect_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.statistics.save_history(HISTORY_PATH)
        ng_count = sum(1 for record in records if record.verdict == "NG")
        ok_count = len(records) - ng_count
        avg_ms = sum(record.elapsed_ms for record in records) / len(records) if records else 0.0
        QMessageBox.information(
            self, "检测完成",
            f"本次检测 {len(records)} 张图片。\n\n"
            f"良品：{ok_count} 张\n不良品：{ng_count} 张\n"
            f"不良率：{(ng_count / len(records) * 100) if records else 0:.1f}%\n"
            f"平均耗时：{avg_ms:.0f} ms/张\n\n"
            f"结果图已保存至 outputs/detections/ 目录。",
        )

    def _on_failed(self, message):
        self.status_bar.showMessage(f"错误：{message}", 6000)

    def _on_history_clicked(self, row, column):
        if row >= len(self.statistics.records):
            return
        record = self.statistics.records[row]
        original = imread(record.source)
        annotated = imread(record.annotated_path)
        self._set_label_image(self.original_label, original)
        if annotated is None:
            self._set_label_image(self.result_label, None)
            self.result_label.setText("历史标注图不可用")
        else:
            self._set_label_image(self.result_label, annotated)
        self.status_bar.showMessage(
            f"{record.filename} · 判定 {record.verdict} · {record.defect_summary()} · {record.elapsed_ms:.0f} ms"
        )

    def _export_report(self):
        if not self.statistics.records:
            QMessageBox.information(self, "提示", "暂无检测记录，无法导出报表。")
            return
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"PCB质检报表_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        path = REPORT_DIR / filename
        try:
            self.statistics.export_excel(path)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        QMessageBox.information(self, "导出成功", f"Excel 质检报表已保存至：\n{path}")

    def _clear_history(self):
        if not self.statistics.records:
            return
        answer = QMessageBox.question(
            self, "确认清空", "确定要清空全部历史检测记录吗？（已保存的图片和报表不会被删除）",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.statistics.clear()
        self._refresh_record_table()
        self._refresh_stats_panel()
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
        self.status_bar.showMessage("历史记录已清空")

    def _show_help(self):
        QMessageBox.information(
            self, "使用说明",
            "1. 点击[上传图片]选择一张或多张 PCB 图像。\n"
            "2. 按需勾选预处理、调整置信度阈值，点击[一键检测]。\n"
            "3. 左侧显示原图与带框结果图，右侧表格记录判定与耗时。\n"
            "4. 点击历史记录行可回看对应图片。\n"
            "5. 点击[导出Excel报表]生成含明细、缺陷统计与汇总的报表。\n\n"
            "检测类别：开路 / 短路 / 针孔 / 毛刺 / 鼠咬 / 伪铜。",
        )

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait()
        self.statistics.save_history(HISTORY_PATH)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
