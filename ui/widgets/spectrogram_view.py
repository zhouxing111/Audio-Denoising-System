"""
ui/widgets/spectrogram_view.py — 频谱图对比组件

使用 matplotlib (嵌入 PySide6) 或 pyqtgraph ImageView 渲染 STFT 频谱图，
左右并排对比带噪音频和降噪音频的时频分布。
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

import librosa
import librosa.display


class SpectrogramView(QWidget):
    """频谱图对比组件。

    并排显示带噪(左)和降噪(右)的 STFT 幅度谱 (dB 尺度)，
    使用 pyqtgraph ImageItem 高性能渲染。
    """

    def __init__(self, parent: QWidget | None = None):
        """初始化频谱图绘图区域。"""
        super().__init__(parent)

        # 左侧: 带噪频谱
        self._noisy_plot = pg.PlotWidget()
        self._noisy_plot.setLabel("bottom", "Time (s)")
        self._noisy_plot.setLabel("left", "Frequency (Hz)")
        self._noisy_plot.setTitle("Noisy Spectrogram")
        self._noisy_img = pg.ImageItem()
        self._noisy_plot.addItem(self._noisy_img)
        self._noisy_colorbar = pg.ColorBarItem(
            values=(-80, 0), colorMap="inferno", label="dB"
        )
        self._noisy_colorbar.setImageItem(self._noisy_img)

        # 右侧: 降噪频谱
        self._denoised_plot = pg.PlotWidget()
        self._denoised_plot.setLabel("bottom", "Time (s)")
        self._denoised_plot.setLabel("left", "Frequency (Hz)")
        self._denoised_plot.setTitle("Denoised Spectrogram")
        self._denoised_img = pg.ImageItem()
        self._denoised_plot.addItem(self._denoised_img)
        self._denoised_colorbar = pg.ColorBarItem(
            values=(-80, 0), colorMap="viridis", label="dB"
        )
        self._denoised_colorbar.setImageItem(self._denoised_img)

        # 布局
        hbox = QHBoxLayout()
        hbox.addWidget(self._noisy_plot)
        hbox.addWidget(self._denoised_plot)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(hbox)
        self._sr = 16000
        self._hop = 256
        self._n_fft = 512

    def set_data(
        self,
        noisy: np.ndarray,
        denoised: np.ndarray,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
    ) -> None:
        """计算并显示两段音频的频谱图。

        Args:
            noisy: 带噪信号.
            denoised: 降噪后信号.
            sr: 采样率.
            n_fft: FFT 点数.
            hop_length: 帧移.
        """
        self._sr = sr
        self._hop = hop_length
        self._n_fft = n_fft

        # 带噪频谱
        noisy_stft = np.abs(librosa.stft(
            noisy.astype(np.float32), n_fft=n_fft, hop_length=hop_length
        ))
        noisy_db = librosa.amplitude_to_db(noisy_stft, ref=np.max, top_db=80)
        self._noisy_img.setImage(noisy_db.T, levels=(-80, 0))
        self._set_axis_transform(self._noisy_plot, self._noisy_img, noisy_db)

        # 降噪频谱
        denoised_stft = np.abs(librosa.stft(
            denoised.astype(np.float32), n_fft=n_fft, hop_length=hop_length
        ))
        denoised_db = librosa.amplitude_to_db(denoised_stft, ref=np.max, top_db=80)
        self._denoised_img.setImage(denoised_db.T, levels=(-80, 0))
        self._set_axis_transform(self._denoised_plot, self._denoised_img, denoised_db)

    def _set_axis_transform(
        self, plot: pg.PlotWidget, img: pg.ImageItem, data: np.ndarray
    ) -> None:
        """设置 ImageItem 的坐标变换，使轴标签对应真实时间和频率。

        Args:
            plot: 目标 PlotWidget.
            img: 目标 ImageItem.
            data: 频谱数据, shape (n_freqs, n_frames).
        """
        n_freqs, n_frames = data.shape
        dur = n_frames * self._hop / self._sr
        tr = pg.QtGui.QTransform()
        tr.scale(dur / n_frames, (self._sr / 2) / n_freqs)
        img.setTransform(tr)
        plot.setXRange(0, dur)
        plot.setYRange(0, self._sr / 2)

    def clear(self) -> None:
        """清空频谱图。"""
        self._noisy_img.clear()
        self._denoised_img.clear()
