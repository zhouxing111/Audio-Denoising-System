"""
models/base.py — 降噪模型抽象基类

定义统一的降噪器接口，所有传统算法和深度学习模型必须继承此类，
强制实现 forward() 和 denoise_audio() 两个核心方法。
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor


class BaseDenoiser(ABC):
    """音频降噪器抽象基类。

    子类必须实现:
    - forward(x): 训练时的张量前向传播.
    - denoise_audio(waveform, sr): 推理时的完整降噪流程 (numpy 进出).
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """训练期间的前向传播，输入/输出均为 PyTorch Tensor。

        Args:
            x: 输入张量，具体形状由子类定义.

        Returns:
            降噪后的张量.
        """
        ...

    @abstractmethod
    def denoise_audio(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """端到端降噪推理，输入/输出均为 numpy 数组。

        Args:
            waveform: 带噪音频波形, shape (n_samples,).
            sr: 采样率 (Hz).

        Returns:
            降噪后的音频波形, shape (n_samples,).
        """
        ...

    @staticmethod
    def _compute_stft(
        waveform: np.ndarray,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
    ) -> np.ndarray:
        """计算短时傅里叶变换 (STFT)，返回复数频谱。

        Args:
            waveform: 输入波形.
            n_fft: FFT 点数.
            hop_length: 帧移.
            win_length: 窗长.

        Returns:
            复数频谱, shape (n_freqs, n_frames).
        """
        import librosa

        return librosa.stft(
            waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )

    @staticmethod
    def _compute_istft(
        stft_matrix: np.ndarray,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        length: int | None = None,
    ) -> np.ndarray:
        """计算逆 STFT，从复数频谱还原波形。

        Args:
            stft_matrix: 复数频谱.
            n_fft: FFT 点数.
            hop_length: 帧移.
            win_length: 窗长.
            length: 输出波形长度 (None 则自动推断).

        Returns:
            还原的时域波形.
        """
        import librosa

        return librosa.istft(
            stft_matrix,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            length=length,
        )
