"""
ui/widgets/waveform_view.py — 时域波形对比组件

使用 pyqtgraph 渲染带噪/降噪双通道波形对比图，
支持缩放、平移和光标数值读取。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget


class WaveformView(QWidget):
    """时域波形对比组件。

    同时绘制带噪(红色)和降噪(绿色)两条波形曲线，
    X 轴为时间 (秒)，Y 轴为归一化幅度。
    """

    def __init__(self, parent: QWidget | None = None):
        """初始化波形绘图区域。"""
        super().__init__(parent)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.setLabel("left", "Amplitude")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.addLegend()

        self._noisy_curve = self._plot.plot(
            [], [], pen=pg.mkPen("#E74C3C", width=1), name="Noisy"
        )
        self._denoised_curve = self._plot.plot(
            [], [], pen=pg.mkPen("#27AE60", width=1), name="Denoised"
        )
        self._clean_curve = self._plot.plot(
            [], [], pen=pg.mkPen("#2980B9", width=1, style=pg.QtCore.Qt.DashLine),
            name="Clean",
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

    def set_data(
        self,
        noisy: np.ndarray,
        denoised: np.ndarray,
        clean: np.ndarray | None = None,
        sr: int = 16000,
    ) -> None:
        """设置波形数据并刷新显示。

        Args:
            noisy: 带噪信号.
            denoised: 降噪后信号.
            clean: 纯净参考信号 (可选).
            sr: 采样率.
        """
        t_noisy = np.arange(len(noisy)) / sr
        t_denoised = np.arange(len(denoised)) / sr

        self._noisy_curve.setData(t_noisy, noisy)
        self._denoised_curve.setData(t_denoised, denoised)

        if clean is not None:
            t_clean = np.arange(len(clean)) / sr
            self._clean_curve.setData(t_clean, clean)
        else:
            self._clean_curve.clear()

        self._plot.autoRange()

    def clear(self) -> None:
        """清空所有波形数据。"""
        self._noisy_curve.clear()
        self._denoised_curve.clear()
        self._clean_curve.clear()
