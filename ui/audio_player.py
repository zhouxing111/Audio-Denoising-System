"""
ui/audio_player.py — 音频回放控制

使用 sounddevice 库实现带噪/降噪音频的同步播放、暂停和切换。
"""

import threading

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt


class AudioPlayer(QWidget):
    """音频回放控制组件。

    支持三段音频切换: 带噪 / 降噪 / 纯净 (如有)。
    提供播放/暂停、进度条拖动功能。
    """

    def __init__(self, parent: QWidget | None = None):
        """初始化音频播放器。"""
        super().__init__(parent)

        self._noisy = None
        self._denoised = None
        self._clean = None
        self._sr = 16000
        self._current_source = "denoised"
        self._playing = False
        self._position = 0
        self._stream = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_position)

        group = QGroupBox("音频回放")

        # 播放控制按钮
        self._btn_play = QPushButton("播放")
        self._btn_play.clicked.connect(self._toggle_play)
        self._btn_stop = QPushButton("停止")
        self._btn_stop.clicked.connect(self._stop)

        # 音源切换
        self._btn_noisy = QPushButton("带噪")
        self._btn_noisy.setCheckable(True)
        self._btn_noisy.clicked.connect(lambda: self._switch_source("noisy"))

        self._btn_denoised = QPushButton("降噪")
        self._btn_denoised.setCheckable(True)
        self._btn_denoised.setChecked(True)
        self._btn_denoised.clicked.connect(lambda: self._switch_source("denoised"))

        self._btn_clean = QPushButton("纯净")
        self._btn_clean.setCheckable(True)
        self._btn_clean.clicked.connect(lambda: self._switch_source("clean"))

        # 进度条
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.sliderMoved.connect(self._seek)
        self._lbl_time = QLabel("00:00 / 00:00")

        # 布局
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_play)
        btn_row.addWidget(self._btn_stop)
        btn_row.addWidget(QLabel("|"))
        btn_row.addWidget(QLabel("音源:"))
        btn_row.addWidget(self._btn_noisy)
        btn_row.addWidget(self._btn_denoised)
        btn_row.addWidget(self._btn_clean)
        btn_row.addStretch()

        slider_row = QHBoxLayout()
        slider_row.addWidget(self._slider)
        slider_row.addWidget(self._lbl_time)

        layout = QVBoxLayout(group)
        layout.addLayout(btn_row)
        layout.addLayout(slider_row)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(group)

    def set_audio(
        self,
        noisy: np.ndarray,
        denoised: np.ndarray,
        clean: np.ndarray | None = None,
        sr: int = 16000,
    ) -> None:
        """设置音频数据。

        Args:
            noisy: 带噪信号.
            denoised: 降噪后信号.
            clean: 纯净参考信号 (可选).
            sr: 采样率.
        """
        self._noisy = noisy.astype(np.float32)
        self._denoised = denoised.astype(np.float32)
        self._clean = clean.astype(np.float32) if clean is not None else None
        self._sr = sr
        self._btn_clean.setEnabled(clean is not None)
        self._position = 0
        self._update_time_label()

    def _get_current_waveform(self) -> np.ndarray | None:
        """获取当前选中音源的波形。"""
        if self._current_source == "noisy":
            return self._noisy
        elif self._current_source == "clean" and self._clean is not None:
            return self._clean
        return self._denoised

    def _switch_source(self, source: str) -> None:
        """切换音源 (保持播放位置)。

        Args:
            source: "noisy", "denoised", "clean".
        """
        self._current_source = source
        for btn, src in [
            (self._btn_noisy, "noisy"),
            (self._btn_denoised, "denoised"),
            (self._btn_clean, "clean"),
        ]:
            btn.setChecked(src == source)

    def _toggle_play(self) -> None:
        """播放/暂停切换。"""
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        """开始播放当前音源。"""
        wf = self._get_current_waveform()
        if wf is None:
            return
        try:
            import sounddevice as sd

            self._playing = True
            self._btn_play.setText("暂停")
            self._stream = sd.OutputStream(
                samplerate=self._sr, channels=1, dtype="float32"
            )
            self._stream.start()
            self._timer.start(50)  # 每 50ms 更新进度
            # 在新线程中写入数据，避免阻塞 UI
            threading.Thread(
                target=self._playback_thread, args=(wf,), daemon=True
            ).start()
        except ImportError:
            self._status_callback and print("sounddevice 未安装")

    def _pause(self) -> None:
        """暂停播放。"""
        self._playing = False
        self._btn_play.setText("播放")
        self._timer.stop()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _stop(self) -> None:
        """停止播放并回到开头。"""
        self._pause()
        self._position = 0
        self._slider.setValue(0)
        self._update_time_label()

    def _playback_thread(self, waveform: np.ndarray) -> None:
        """后台播放线程，流式写入音频数据。

        Args:
            waveform: 要播放的波形数据.
        """
        chunk_size = self._sr // 10  # 100ms 块
        pos = self._position
        while self._playing and pos < len(waveform):
            end = min(pos + chunk_size, len(waveform))
            chunk = waveform[pos:end]
            if self._stream:
                self._stream.write(chunk)
            pos = end
            self._position = pos
            if pos >= len(waveform):
                self._playing = False
                break
        self._playing = False
        if self._position >= len(waveform):
            self._position = 0

    def _update_position(self) -> None:
        """定时器回调：更新进度条和时长标签。"""
        wf = self._get_current_waveform()
        if wf is None:
            return
        total = len(wf)
        if total > 0:
            self._slider.setValue(int(self._position / total * 1000))
        self._update_time_label()

    def _seek(self, slider_val: int) -> None:
        """拖动进度条跳转播放位置。

        Args:
            slider_val: 滑块值 (0~1000).
        """
        wf = self._get_current_waveform()
        if wf is None:
            return
        self._position = int(len(wf) * slider_val / 1000)
        self._update_time_label()

    def _update_time_label(self) -> None:
        """更新时长标签格式: MM:SS / MM:SS。"""
        wf = self._get_current_waveform()
        if wf is None:
            self._lbl_time.setText("00:00 / 00:00")
            return
        total_s = len(wf) / self._sr
        pos_s = self._position / self._sr
        self._lbl_time.setText(
            f"{_fmt_time(pos_s)} / {_fmt_time(total_s)}"
        )


def _fmt_time(seconds: float) -> str:
    """将秒数格式化为 MM:SS。

    Args:
        seconds: 秒数.

    Returns:
        MM:SS 格式字符串.
    """
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"
