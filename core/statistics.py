"""质检统计与 Excel 报表导出。"""
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from config import CLASS_NAMES_CN


@dataclass
class DetectionRecord:
    timestamp: str
    filename: str
    source: str
    verdict: str
    top_confidence: float
    defects: list = field(default_factory=list)
    elapsed_ms: float = 0.0
    annotated_path: str = ""

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def defect_summary(self):
        counter = Counter(defect["class_name"] for defect in self.defects)
        return "、".join(f"{name}×{count}" for name, count in counter.items()) if counter else "无"


class QualityStatistics:
    def __init__(self):
        self.records = []

    def add(self, record):
        self.records.append(record)

    def clear(self):
        self.records.clear()

    def summary(self):
        total = len(self.records)
        ng = sum(1 for record in self.records if record.verdict == "NG")
        elapsed = [record.elapsed_ms for record in self.records]
        return {
            "total": total,
            "ok": total - ng,
            "ng": ng,
            "ng_rate": (ng / total * 100) if total else 0.0,
            "avg_ms": (sum(elapsed) / len(elapsed)) if elapsed else 0.0,
        }

    def defect_counts(self):
        counter = Counter()
        for record in self.records:
            counter.update(defect["class_name"] for defect in record.defects)
        return counter

    def save_history(self, path):
        Path(path).write_text(
            json.dumps([asdict(record) for record in self.records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_history(self, path):
        path = Path(path)
        if not path.exists():
            return
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
            self.records = [DetectionRecord.from_dict(item) for item in items]
        except (json.JSONDecodeError, TypeError, ValueError):
            self.records = []

    @staticmethod
    def _style_header(worksheet, columns):
        fill = PatternFill("solid", fgColor="2F6FDB")
        for col, width in enumerate(columns, start=1):
            cell = worksheet.cell(row=1, column=col)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
            worksheet.column_dimensions[get_column_letter(col)].width = width
        worksheet.freeze_panes = "A2"

    def export_excel(self, path):
        """导出三张工作表：检测明细 / 缺陷统计 / 质检汇总。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        workbook = Workbook()

        detail = workbook.active
        detail.title = "检测明细"
        detail.append(["序号", "检测时间", "图片文件", "判定", "最高置信度",
                       "缺陷数", "缺陷类别", "耗时(ms)", "标注图路径"])
        for index, record in enumerate(self.records, start=1):
            detail.append([
                index, record.timestamp, record.filename, record.verdict,
                f"{record.top_confidence:.4f}", len(record.defects),
                record.defect_summary(), f"{record.elapsed_ms:.1f}", record.annotated_path,
            ])
        self._style_header(detail, [7, 21, 26, 9, 14, 9, 32, 12, 46])
        for row in detail.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="center", wrap_text=True)
            verdict_cell = row[3]
            verdict_cell.font = Font(bold=True, color="C00000" if verdict_cell.value == "NG" else "008000")

        stat = workbook.create_sheet("缺陷统计")
        stat.append(["序号", "缺陷类别", "出现次数", "占比(%)"])
        counts = self.defect_counts()
        total_boxes = sum(counts.values())
        for index, name in enumerate(CLASS_NAMES_CN, start=1):
            count = counts.get(name, 0)
            stat.append([index, name, count, f"{(count / total_boxes * 100):.2f}" if total_boxes else "0.00"])
        self._style_header(stat, [8, 18, 14, 14])

        overview = workbook.create_sheet("质检汇总")
        overview.append(["指标", "数值"])
        summary = self.summary()
        overview.append(["检测总数", summary["total"]])
        overview.append(["良品数量", summary["ok"]])
        overview.append(["不良品数量", summary["ng"]])
        overview.append(["不良率(%)", f"{summary['ng_rate']:.2f}"])
        overview.append(["平均单张耗时(ms)", f"{summary['avg_ms']:.1f}"])
        overview.append(["报表生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        self._style_header(overview, [24, 30])

        workbook.save(path)
        return path
