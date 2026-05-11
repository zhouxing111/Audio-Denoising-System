"""
models/wiener.py — 频域维纳滤波降噪

基于局部 SNR 估计的经典维纳滤波器。
使用前 N 帧作为纯噪声段估计噪声功率谱，后续帧通过维纳增益衰减噪声。
"""

import numpy as np
from numpy.fft import rfft, irfft


class WienerFilter:
    """频域维纳滤波降噪器。

    使用重叠帧处理，对每帧短时傅里叶数据应用维纳增益函数，
    将噪声主导的频段能量衰减至接近零，保留语音主导的频段。

    使用示例:
        wf = WienerFilter(frame_ms=32, noise_window_ms=500)
        denoised = wf.denoise_audio(noisy_waveform, sr=16000)
    """

    def __init__(
        self,
        frame_ms: float = 32.0,
        noise_window_ms: float = 500.0,
        noise_threshold: float = 0.05,
    ):
        """初始化维纳滤波器参数。

        Args:
            frame_ms: 每帧时长 (ms).
            noise_window_ms: 噪声估计窗口 (ms), 使用音频开头这段时间估计噪声.
            noise_threshold: 噪声门限 (相对能量), 低于此值的频段进一步抑制.
        """
        self.frame_ms = frame_ms
        self.noise_window_ms = noise_window_ms
        self.noise_threshold = noise_threshold

    def denoise_audio(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """对音频执行维纳滤波降噪。

        Args:
            waveform: 带噪音频波形, shape (n_samples,).
            sr: 采样率 (Hz).

        Returns:
            降噪后波形, shape (n_samples,).
        """
        frame_len = int(sr * self.frame_ms / 1000.0)
        hop = frame_len // 2
        noise_frames = int(self.noise_window_ms / self.frame_ms)

        # 分帧 (使用汉宁窗)
        frames = self._frame_signal(waveform, frame_len, hop)
        n_frames, n_fft = frames.shape

        # 对每帧做 FFT
        spec = rfft(frames, axis=1)  # (n_frames, n_freqs)

        # 用前 noise_frames 帧估计噪声功率谱
        noise_psd = np.mean(np.abs(spec[:noise_frames]) ** 2, axis=0)  # (n_freqs,)

        # 维纳增益
        denoised_spec = np.zeros_like(spec, dtype=np.complex128)
        for i in range(n_frames):
            signal_psd = np.abs(spec[i]) ** 2
            # Wiener gain: G = max(0, (S - N) / S)
            wiener_gain = np.maximum(0.0, 1.0 - noise_psd / (signal_psd + 1e-12))
            # 额外门限抑制
            wiener_gain[wiener_gain < self.noise_threshold] = 0.0
            denoised_spec[i] = spec[i] * wiener_gain

        # 逆 FFT + 重叠相加
        denoised_frames = irfft(denoised_spec, n=n_fft, axis=1)
        return self._overlap_add(denoised_frames, hop, len(waveform))

    def _frame_signal(
        self, waveform: np.ndarray, frame_len: int, hop: int
    ) -> np.ndarray:
        """将一维信号分割为重叠帧。

        Args:
            waveform: 输入波形.
            frame_len: 帧长 (采样点).
            hop: 帧移 (采样点).

        Returns:
            分帧矩阵, shape (n_frames, frame_len).
        """
        n = len(waveform)
        n_frames = max(1, (n - frame_len) // hop + 1)
        frames = np.zeros((n_frames, frame_len), dtype=np.float64)
        window = np.hanning(frame_len)
        for i in range(n_frames):
            start = i * hop
            end = start + frame_len
            segment = waveform[start:end].astype(np.float64)
            frames[i] = segment * window
        return frames

    def _overlap_add(
        self, frames: np.ndarray, hop: int, target_len: int
    ) -> np.ndarray:
        """重叠相加还原波形。

        Args:
            frames: 帧矩阵, shape (n_frames, frame_len).
            hop: 帧移.
            target_len: 目标输出长度.

        Returns:
            还原的波形.
        """
        n_frames, frame_len = frames.shape
        output = np.zeros(target_len, dtype=np.float64)
        window = np.hanning(frame_len)
        win_sum = np.zeros(target_len, dtype=np.float64)
        for i in range(n_frames):
            start = i * hop
            end = start + frame_len
            output[start:end] += frames[i] * window
            win_sum[start:end] += window**2
        win_sum[win_sum < 1e-12] = 1.0
        return (output / win_sum).astype(np.float32)
