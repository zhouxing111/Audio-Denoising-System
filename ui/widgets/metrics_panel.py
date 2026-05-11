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

    def clear(self) -> None:
        """清空指标表格。"""
        self._table.setRowCount(0)
