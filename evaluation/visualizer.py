"""
evaluation/visualizer.py — 音频可视化工具

提供时域波形、STFT 频谱图、梅尔谱的绘制函数。
接受 numpy 波形数组，返回 matplotlib Figure 对象，可直接嵌入 GUI 或保存为 PNG。
"""

import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")  # 非交互后端，可在无 GUI 环境运行
import matplotlib.pyplot as plt

# 统一配色方案: noisy=红色系, clean=蓝色系, denoised=绿色系
COLORS = {
    "noisy": "#E74C3C",
    "clean": "#2980B9",
    "denoised": "#27AE60",
}


def plot_waveform(
    noisy: np.ndarray,
    denoised: np.ndarray,
    clean: np.ndarray | None = None,
    sr: int = 16000,
    title: str = "Waveform Comparison",
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """绘制时域波形对比图。

    Args:
        noisy: 带噪信号.
        denoised: 降噪后信号.
        clean: 纯净参考信号 (可选).
        sr: 采样率.
        title: 图表标题.
        figsize: 图表尺寸.

    Returns:
        matplotlib Figure 对象.
    """
    fig, ax = plt.subplots(figsize=figsize)
    t = np.arange(len(noisy)) / sr

    ax.plot(t, noisy, color=COLORS["noisy"], alpha=0.6, linewidth=0.5, label="Noisy")
    ax.plot(t, denoised, color=COLORS["denoised"], alpha=0.8, linewidth=0.5, label="Denoised")
    if clean is not None:
        min_len = min(len(t), len(clean))
        ax.plot(
            t[:min_len], clean[:min_len], color=COLORS["clean"],
            alpha=0.5, linewidth=0.5, label="Clean"
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, t[-1])
    plt.tight_layout()
    return fig


def plot_spectrogram(
    waveform: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    title: str = "STFT Spectrogram",
    figsize: tuple[int, int] = (6, 4),
    color: str = "viridis",
) -> plt.Figure:
    """绘制 STFT 频谱图 (dB 尺度)。

    Args:
        waveform: 输入波形.
        sr: 采样率.
        n_fft: FFT 点数.
        hop_length: 帧移.
        title: 图表标题.
        figsize: 图表尺寸.
        color: matplotlib colormap 名称.

    Returns:
        matplotlib Figure 对象.
    """
    import librosa
    import librosa.display

    stft = librosa.stft(waveform.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    mag_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        mag_db, sr=sr, hop_length=hop_length, x_axis="time",
        y_axis="hz", ax=ax, cmap=color,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 128,
    title: str = "Mel Spectrogram",
    figsize: tuple[int, int] = (6, 4),
    color: str = "magma",
) -> plt.Figure:
    """绘制梅尔频谱图。

    Args:
        waveform: 输入波形.
        sr: 采样率.
        n_fft: FFT 点数.
        hop_length: 帧移.
        n_mels: 梅尔滤波器组数量.
        title: 图表标题.
        figsize: 图表尺寸.
        color: matplotlib colormap 名称.

    Returns:
        matplotlib Figure 对象.
    """
    import librosa
    import librosa.display

    mel = librosa.feature.melspectrogram(
        y=waveform.astype(np.float32), sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop_length, x_axis="time",
        y_axis="mel", ax=ax, cmap=color,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_comparison(
    noisy: np.ndarray,
    denoised: np.ndarray,
    clean: np.ndarray | None = None,
    sr: int = 16000,
) -> plt.Figure:
    """绘制完整对比图 — 上方波形，下方频谱。

    Args:
        noisy: 带噪信号.
        denoised: 降噪后信号.
        clean: 纯净参考信号 (可选).
        sr: 采样率.

    Returns:
        组合 matplotlib Figure 对象.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 波形
    t = np.arange(len(noisy)) / sr
    ax_wf = axes[0, 0]
    ax_wf.plot(t, noisy, color=COLORS["noisy"], alpha=0.6, linewidth=0.4, label="Noisy")
    ax_wf.plot(t, denoised, color=COLORS["denoised"], alpha=0.8, linewidth=0.4, label="Denoised")
    ax_wf.set_title("Waveform")
    ax_wf.set_xlabel("Time (s)")
    ax_wf.legend(fontsize=7)
    ax_wf.set_xlim(0, t[-1])

    # 带噪频谱
    import librosa
    import librosa.display

    ax_spec_n = axes[1, 0]
    stft_n = librosa.stft(noisy.astype(np.float32), n_fft=512, hop_length=256)
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(stft_n), ref=np.max),
        sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax_spec_n,
        cmap="inferno",
    )
    ax_spec_n.set_title("Noisy Spectrogram")

    # 降噪频谱
    ax_spec_d = axes[1, 1]
    stft_d = librosa.stft(denoised.astype(np.float32), n_fft=512, hop_length=256)
    librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(stft_d), ref=np.max),
        sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax_spec_d,
        cmap="viridis",
    )
    ax_spec_d.set_title("Denoised Spectrogram")

    # 纯净频谱 (如有)
    if clean is not None:
        ax_spec_c = axes[0, 1]
        stft_c = librosa.stft(clean[:len(noisy)].astype(np.float32), n_fft=512, hop_length=256)
        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(stft_c), ref=np.max),
            sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax_spec_c,
            cmap="Blues",
        )
        ax_spec_c.set_title("Clean Spectrogram")
    else:
        axes[0, 1].axis("off")

    plt.tight_layout()
    return fig
