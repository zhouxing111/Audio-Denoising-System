"""
ui/widgets/diagnosis_panel.py — 噪声诊断结果面板

展示噪声类型诊断结论、频段能量占比和频谱图热力图。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


class DiagnosisPanel(QWidget):
    """噪声诊断结果面板。

    左侧文本展示诊断结论和频段占比，
    右侧显示噪声平均频谱曲线。
    """

    def __init__(self, parent: QWidget | None = None):
        """初始化诊断面板。"""
        super().__init__(parent)

        group = QGroupBox("噪声类型诊断")

        # 左侧: 文本结论
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumWidth(350)
        self._text.setPlaceholderText("执行降噪后自动展示噪声诊断结果...")

        # 右侧: 频谱图
        self._spec_plot = pg.PlotWidget()
        self._spec_plot.setLabel("bottom", "Frequency (Hz)")
        self._spec_plot.setLabel("left", "Normalized Magnitude")
        self._spec_plot.setTitle("Noise Spectrum Profile")
        self._spec_plot.showGrid(x=True, y=True, alpha=0.3)
        self._spec_curve = self._spec_plot.plot(
            [], [], pen=pg.mkPen("#E74C3C", width=2)
        )

        # 频段标注竖线
        self._band_lines = []
        for hz, color in [(500, "#F39C12"), (3000, "#2980B9")]:
            line = pg.InfiniteLine(
                pos=hz, angle=90, pen=pg.mkPen(color, style=pg.QtCore.Qt.DashLine)
            )
            self._spec_plot.addItem(line)
            self._band_lines.append(line)

        hbox = QHBoxLayout()
        hbox.addWidget(self._text)
        hbox.addWidget(self._spec_plot, stretch=1)

        layout = QVBoxLayout(group)
        layout.addLayout(hbox)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(group)

    def set_diagnosis(
        self,
        noise_type_key: str,
        noise_type_label: str,
        details: dict,
        spectrum_profile: dict,
    ) -> None:
        """展示诊断结果。

        Args:
            noise_type_key: 噪声类型标识.
            noise_type_label: 噪声类型中文标签.
            details: 频段占比等详细数据.
            spectrum_profile: 完整频谱数据 (含 freqs, noise_spectrum).
        """
        # 文本区域
        lines = [
            f"诊断结论: {noise_type_label}",
            "-" * 45,
            "频段能量占比:",
            f"  低频 (<500Hz):    {details.get('low_ratio', 0)*100:5.1f}%",
            f"  中频 (0.5-3kHz):  {details.get('mid_ratio', 0)*100:5.1f}%",
            f"  高频 (>3kHz):     {details.get('high_ratio', 0)*100:5.1f}%",
            f"  工频谐波强度:      {details.get('harmonic_strength', 0):.4f}",
            f"  主导频率:          {details.get('dominant_freq_hz', 0):.1f} Hz",
            "-" * 45,
            f"噪声帧占比: {details.get('noise_frame_ratio', 0)*100:.1f}% "
            f"({details.get('noise_frames', 0)}/{details.get('total_frames', 0)})",
            f"频谱平坦度: {details.get('spectral_flatness', 0):.6f}",
        ]
        self._text.setPlainText("\n".join(lines))

        # 频谱曲线
        freqs = spectrum_profile.get("freqs", np.linspace(0, 8000, 257))
        spec = spectrum_profile.get("noise_spectrum", np.zeros_like(freqs))
        self._spec_curve.setData(freqs, spec)
        self._spec_plot.autoRange()

    def clear(self) -> None:
        """清空诊断面板。"""
        self._text.clear()
        self._spec_curve.clear()
