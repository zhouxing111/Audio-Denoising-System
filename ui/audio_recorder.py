"""
ui/audio_recorder.py — 麦克风在线录音组件

使用 sounddevice.InputStream 从系统默认麦克风实时采集音频，
提供实时波形预览、时长控制、开始/停止/重录功能。
录音完成后通过信号将波形传递给主窗口进行降噪处理。
"""

import threading

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Signal, Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class RingBuffer:
    """线程安全的环形缓冲区，用于音频回调线程写入、主线程读取。

    sounddevice InputStream 的音频回调在独立线程运行，
    回调只做快速 memcpy 写入此缓冲区，主线程通过 QTimer 轮询读取。
    """

    def __init__(self, max_samples: int):
        """初始化环形缓冲区。

        Args:
            max_samples: 最大容量 (采样点数)，即 max_duration * sample_rate.
        """
        self._buffer = np.zeros(max_samples, dtype=np.float32)
        self._capacity = max_samples
        self._write_pos = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        """写入音频数据（由音频回调线程调用）。

        如果缓冲区满则丢弃超出部分，保证不会越界。

        Args:
            data: 音频数据块, shape (n_samples,), float32.
        """
        n = len(data)
        with self._lock:
            if self._write_pos + n <= self._capacity:
                self._buffer[self._write_pos : self._write_pos + n] = data
                self._write_pos += n

    def get_new_samples(self, start: int) -> tuple[np.ndarray, int]:
        """从 start 位置读取到当前写入位置的全部新数据。

        Args:
            start: 上次读取到的位置 (采样点索引).

        Returns:
            (new_data, current_write_pos): 新采样数据和当前写入位置.
        """
        with self._lock:
            end = self._write_pos
            if end <= start:
                return np.array([], dtype=np.float32), end
            return self._buffer[start:end].copy(), end

    @property
    def total_samples(self) -> int:
        """已写入的总采样点数。"""
        with self._lock:
            return self._write_pos


class AudioRecorder(QWidget):
    """麦克风在线录音组件。

    信号:
        recording_finished(waveform, sr): 录音完成时携带完整波形和采样率.
        recording_error(error_msg): 录音出错时携带错误消息.
    """

    recording_finished = Signal(np.ndarray, int)
    recording_error = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        """初始化录音组件。"""
        super().__init__(parent)
        self._sr = 16000
        self._max_duration = 30.0  # 最长录音 30 秒
        self._buffer: RingBuffer | None = None
        self._stream = None
        self._recording = False
        self._elapsed = 0.0
        self._last_read = 0
        self._total_recorded = 0

        # UI timer: 每 50ms 更新实时波形和时长
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_live)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """构建录音组件 UI 布局。"""
        group = QGroupBox("在线录音")
        layout = QVBoxLayout(group)

        # --- 状态与时长 ---
        info_row = QHBoxLayout()
        self._status_label = QLabel("● 就绪")
        self._status_label.setStyleSheet("color: gray; font-weight: bold;")
        info_row.addWidget(self._status_label)

        self._time_label = QLabel("00:00.0 / 00:30.0")
        info_row.addWidget(self._time_label)
        info_row.addStretch()
        layout.addLayout(info_row)

        # --- 进度条 ---
        self._progress = QProgressBar()
        self._progress.setRange(0, 30000)  # ms
        self._progress.setValue(0)
        self._progress.setVisible(True)
        layout.addWidget(self._progress)

        # --- 控制按钮 ---
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("开始录音")
        self._btn_start.clicked.connect(self.start_recording)
        btn_row.addWidget(self._btn_start)

        self._btn_stop = QPushButton("停止录音")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self.stop_recording)
        btn_row.addWidget(self._btn_stop)

        self._btn_retry = QPushButton("重新录制")
        self._btn_retry.setEnabled(False)
        self._btn_retry.clicked.connect(self.reset_and_restart)
        btn_row.addWidget(self._btn_retry)
        layout.addLayout(btn_row)

        # --- 实时波形预览 ---
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.setLabel("left", "Amplitude")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setYRange(-1.0, 1.0)
        self._plot.setTitle("实时波形预览")
        self._live_curve = self._plot.plot(
            [], [], pen=pg.mkPen("#E74C3C", width=1.5)
        )
        layout.addWidget(self._plot, stretch=2)

        # --- 参数信息 ---
        info_label = QLabel(
            f"采样率: {self._sr} Hz | 单声道 | float32 | 最长 {self._max_duration:.0f}s"
        )
        info_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(info_label)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(group)

    # ---------- 公共 API ----------

    def start_recording(self) -> None:
        """开始录音。

        打开 sounddevice.InputStream，音频回调将数据写入 RingBuffer，
        QTimer 每 50ms 更新波形预览和时长。
        """
        try:
            import sounddevice as sd
        except ImportError:
            self.recording_error.emit("sounddevice 未安装 (pip install sounddevice)")
            return

        # 初始化缓冲区
        self._buffer = RingBuffer(int(self._max_duration * self._sr))
        self._last_read = 0
        self._elapsed = 0.0
        self._total_recorded = 0

        try:
            self._stream = sd.InputStream(
                samplerate=self._sr,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            self.recording_error.emit(f"无法打开麦克风: {e}")
            return

        self._recording = True
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_retry.setEnabled(False)
        self._status_label.setText("● 录音中...")
        self._status_label.setStyleSheet("color: #E74C3C; font-weight: bold;")
        self._progress.setValue(0)
        self._timer.start(50)

    def stop_recording(self) -> None:
        """停止录音，关闭流，发射 finished 信号携带完整波形。"""
        self._end_recording()
        if self._buffer is None or self._buffer.total_samples == 0:
            return
        total = self._buffer.total_samples
        waveform, _ = self._buffer.get_new_samples(0)
        # 截取实际录制的部分
        waveform = waveform[:total]
        self._total_recorded = total
        self._btn_retry.setEnabled(True)
        self.recording_finished.emit(waveform.astype(np.float32), self._sr)

    def reset_and_restart(self) -> None:
        """清除当前录音缓冲区，重新开始录音。"""
        self._live_curve.clear()
        self._progress.setValue(0)
        self._time_label.setText(
            f"00:00.0 / {int(self._max_duration):02d}:00.0"
        )
        self._total_recorded = 0
        self.start_recording()

    def get_recorded_audio(self) -> tuple[np.ndarray, int] | None:
        """获取录制的音频数据（供主窗口在 recording_finished 之外随时查询）。

        Returns:
            (waveform, sr) 或 None (如果未录制).
        """
        if self._buffer is None:
            return None
        total = self._buffer.total_samples
        if total == 0:
            return None
        waveform, _ = self._buffer.get_new_samples(0)
        return waveform[:total].astype(np.float32), self._sr

    # ---------- 内部方法 ----------

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """sounddevice InputStream 的音频回调（运行在独立线程）。

        只做快速写入 RingBuffer，不涉及任何 UI 操作。

        Args:
            indata: 输入音频数据, shape (frames, channels).
            frames: 帧数.
            time_info: sounddevice 时间戳 (未使用).
            status: 流状态标志.
        """
        if status:
            # 输入溢出等警告，静默忽略
            pass
        if self._buffer is not None:
            # 提取单声道 → 写入环形缓冲区
            self._buffer.write(indata[:, 0].astype(np.float32))

    def _update_live(self) -> None:
        """QTimer 回调 (每 50ms)：更新波形预览和时长。

        从 RingBuffer 拉取新数据 → 追加到波形曲线 → 刷新进度条。
        """
        if not self._recording or self._buffer is None:
            return

        # 检查是否到达最大时长
        self._elapsed += 0.05
        self._progress.setValue(int(self._elapsed * 1000))
        self._time_label.setText(
            f"{_fmt_time(self._elapsed)} / {_fmt_time(self._max_duration)}"
        )

        if self._elapsed >= self._max_duration:
            self._end_recording()
            # 自动停止后也发射信号
            self._btn_retry.setEnabled(True)
            total = self._buffer.total_samples
            waveform, _ = self._buffer.get_new_samples(0)
            waveform = waveform[:total]
            self._total_recorded = total
            self.recording_finished.emit(waveform.astype(np.float32), self._sr)
            return

        # 拉取新数据更新波形预览 (显示最近 2 秒)
        new_data, self._last_read = self._buffer.get_new_samples(self._last_read)
        show_window = int(2.0 * self._sr)
        total = self._buffer.total_samples
        start_idx = max(0, total - show_window)
        preview = self._buffer._buffer[start_idx:total]
        t = np.arange(len(preview)) / self._sr
        self._live_curve.setData(t, preview)

    def _end_recording(self) -> None:
        """内部：关闭流和定时器，设置 UI 为非录音状态。"""
        self._recording = False
        self._timer.stop()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status_label.setText("● 录音完成")
        self._status_label.setStyleSheet("color: #27AE60; font-weight: bold;")


def _fmt_time(seconds: float) -> str:
    """将秒数格式化为 MM:SS.m 格式。

    Args:
        seconds: 秒数.

    Returns:
        MM:SS.m 格式的时间字符串.
    """
    m = int(seconds) // 60
    s = seconds % 60
    return f"{m:02d}:{s:04.1f}"
