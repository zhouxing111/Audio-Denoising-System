"""
models/spectral_sub.py — 谱减法降噪

经典的频域降噪算法。从带噪幅度谱中减去估计的噪声幅度谱，
配合过减因子和频谱底板，避免音乐噪声伪影。
"""

import numpy as np
from numpy.fft import rfft, irfft


class SpectralSubtraction:
    """谱减法降噪器。

    原理: |X_clean| = max(|X_noisy| - α*|N_est|, β*|X_noisy|)
    其中 α 为过减因子 (控制去噪力度), β 为频谱底板 (避免过度衰减).

    使用示例:
        ss = SpectralSubtraction(alpha=2.0, beta=0.01, frame_ms=32)
        denoised = ss.denoise_audio(noisy_waveform, sr=16000)
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 0.01,
        frame_ms: float = 32.0,
    ):
        """初始化谱减法参数。

        Args:
            alpha: 过减因子 (>1 增强去噪力度, 典型范围 1.5~4.0).
            beta: 频谱底板 (0~1), 防止过度衰减为 0.
            frame_ms: 帧长 (ms).
        """
        self.alpha = alpha
        self.beta = beta
        self.frame_ms = frame_ms

    def denoise_audio(
        self, waveform: np.ndarray, sr: int, noise_frames: int = 10
    ) -> np.ndarray:
        """谱减法降噪。

        前 noise_frames 帧作为纯噪声段来估计噪声幅度谱，
        后续对所有帧的幅度谱执行谱减。

        Args:
            waveform: 带噪音频波形.
            sr: 采样率 (Hz).
            noise_frames: 用于估计噪声的前 N 帧数.

        Returns:
            降噪后波形.
        """
        frame_len = int(sr * self.frame_ms / 1000.0)
        hop = frame_len // 2

        frames = self._frame_signal(waveform, frame_len, hop)
        n_frames, n_fft = frames.shape

        # FFT
        spec = rfft(frames, axis=1)
        mag = np.abs(spec)
        phase = np.angle(spec)

        # 噪声幅度谱估计 (前 noise_frames 帧)
        noise_mag = np.mean(mag[:noise_frames], axis=0)

        # 谱减: clean_mag = max(mag - alpha*noise, beta*mag)
        clean_mag = np.maximum(mag - self.alpha * noise_mag, self.beta * mag)

        # 重建复数谱 (用原始相位)
        clean_spec = clean_mag * np.exp(1j * phase)

        # 逆 FFT + 重叠相加
        denoised_frames = irfft(clean_spec, n=n_fft, axis=1)
        return self._overlap_add(denoised_frames, hop, len(waveform))

    def _frame_signal(
        self, waveform: np.ndarray, frame_len: int, hop: int
    ) -> np.ndarray:
        """分帧加窗。"""
        n = len(waveform)
        n_frames = max(1, (n - frame_len) // hop + 1)
        frames = np.zeros((n_frames, frame_len), dtype=np.float64)
        window = np.hanning(frame_len)
        for i in range(n_frames):
            start = i * hop
            frames[i] = (
                waveform[start : start + frame_len].astype(np.float64) * window
            )
        return frames

    def _overlap_add(
        self, frames: np.ndarray, hop: int, target_len: int
    ) -> np.ndarray:
        """重叠相加还原波形。"""
        n_frames, frame_len = frames.shape
        output = np.zeros(target_len, dtype=np.float64)
        window = np.hanning(frame_len)
        win_sum = np.zeros(target_len, dtype=np.float64)
        for i in range(n_frames):
            start = i * hop
            output[start : start + frame_len] += frames[i] * window
            win_sum[start : start + frame_len] += window**2
        win_sum[win_sum < 1e-12] = 1.0
        return (output / win_sum).astype(np.float32)
