"""
ui/widgets/metrics_panel.py — 评估指标得分板

以表格形式展示所有客观评估指标，支持颜色编码
(绿色=优秀, 黄色=一般, 红色=较差)。
"""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MetricsPanel(QWidget):
    """评估指标得分板。

    以 QTableWidget 表格形式展示所有客观评估指标，
    根据经验阈值自动着色。
    """

    # 各指标的优秀/一般/较差阈值 (值越大越好)
    THRESHOLDS = {
        "SNR (dB)": (15, 5),
        "SegSNR (dB)": (10, 0),
        "SI-SDR (dB)": (15, 5),
        "STOI": (0.85, 0.65),
        "PESQ_WB": (3.5, 2.5),
        "PESQ_NB": (3.0, 2.0),
        "LSD (dB)": (3.0, 6.0),
    }

    def __init__(self, parent: QWidget | None = None):
        """初始化得分板表格。"""
        super().__init__(parent)

        group = QGroupBox("评估指标得分板")
        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["指标", "数值"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)

        layout = QVBoxLayout(group)
        layout.addWidget(self._table)
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(group)

    def set_metrics(self, metrics: dict) -> None:
        """填充指标数据并着色。

        Args:
            metrics: compute_all_metrics 返回的指标字典.
        """
        self._table.setRowCount(len(metrics))
        for row, (name, value) in enumerate(metrics.items()):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)

            if isinstance(value, float) and not np.isnan(value):
                val_str = f"{value:.4f}"
                val_item = QTableWidgetItem(val_str)
                val_item.setTextAlignment(Qt.AlignCenter)
                self._apply_color(val_item, name, value)
            else:
                val_item = QTableWidgetItem("N/A")
                val_item.setForeground(Qt.gray)

            val_item.setFlags(val_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, val_item)

    def _apply_color(
        self, item: QTableWidgetItem, metric_name: str, value: float
    ) -> None:
        """根据阈值给指标值着色。

        Args:
            item: 表格项.
            metric_name: 指标名称.
            value: 指标数值.
        """
        thresholds = self.THRESHOLDS.get(metric_name)
        if thresholds is None:
            return
        good, bad = thresholds

        # LSD 越小越好，其他越大越好
        if metric_name == "LSD (dB)":
            if value < good:
                item.setForeground(Qt.darkGreen)
            elif value > bad:
                item.setForeground(Qt.red)
            else:
                item.setForeground(Qt.darkYellow)
        else:
            if value > good:
                item.setForeground(Qt.darkGreen)
            elif value < bad:
                item.setForeground(Qt.red)
            else:
                item.setForeground(Qt.darkYellow)

    def set_comparison(self, all_metrics: dict[str, dict[str, float]]) -> None:
        """对比模式：行=方法，列=指标。

        Args:
            all_metrics: {"Wiener Filter": {"SNR": 12.3, "PESQ": 3.1}, ...}.
        """
        method_names = list(all_metrics.keys())
        if not method_names:
            return
        # 收集所有指标名
        metric_names = list(next(iter(all_metrics.values())).keys())
        # 表格: 行=方法, 列=指标
        self._table.setRowCount(len(method_names))
        self._table.setColumnCount(len(metric_names) + 1)
        self._table.setHorizontalHeaderLabels(["Method"] + metric_names)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for j in range(1, len(metric_names) + 1):
            self._table.horizontalHeader().setSectionResizeMode(j, QHeaderView.Stretch)

        for i, method in enumerate(method_names):
            name_item = QTableWidgetItem(method)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 0, name_item)
            for j, mname in enumerate(metric_names):
                val = all_metrics[method].get(mname, float("nan"))
                if not np.isnan(val):
                    val_item = QTableWidgetItem(f"{val:.4f}")
                    val_item.setTextAlignment(Qt.AlignCenter)
                    self._apply_color(val_item, mname, val)
                else:
                    val_item = QTableWidgetItem("N/A")
                val_item.setFlags(val_item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(i, j + 1, val_item)

    def set_before_after(
        self, metrics_before: dict[str, float], metrics_after: dict[str, float],
    ) -> None:
        """前后对比模式：列 = Metric | Before | After | Improvement。

        Args:
            metrics_before: 降噪前指标 (clean vs noisy).
            metrics_after: 降噪后指标 (clean vs denoised).
        """
        metric_names = list(metrics_before.keys())
        if not metric_names:
            return
        self._table.setRowCount(len(metric_names))
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Metric", "Before", "After", "Improvement"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for j in range(1, 4):
            self._table.horizontalHeader().setSectionResizeMode(j, QHeaderView.ResizeToContents)

        for i, name in enumerate(metric_names):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 0, name_item)

            before_val = metrics_before.get(name, float("nan"))
            after_val = metrics_after.get(name, float("nan"))

            for col, val in [(1, before_val), (2, after_val)]:
                if not np.isnan(val):
                    item = QTableWidgetItem(f"{val:.4f}")
                    item.setTextAlignment(Qt.AlignCenter)
                    self._apply_color(item, name, val)
                else:
                    item = QTableWidgetItem("N/A")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(i, col, item)

            # Improvement 列
            if not np.isnan(before_val) and not np.isnan(after_val):
                diff = after_val - before_val
                # LSD 越小越好，所以 improvement 符号反转
                if name == "LSD (dB)":
                    diff = -diff
                diff_item = QTableWidgetItem(f"{diff:+.4f}")
                diff_item.setTextAlignment(Qt.AlignCenter)
                diff_item.setForeground(Qt.darkGreen if diff > 0 else (Qt.red if diff < 0 else Qt.gray))
            else:
                diff_item = QTableWidgetItem("N/A")
            diff_item.setFlags(diff_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 3, diff_item)

    def clear(self) -> None:
        """清空指标表格。"""
        self._table.setRowCount(0)
