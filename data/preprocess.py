"""
data/preprocess.py — 音频预处理管线

提供音频加载、重采样、幅值归一化和 RIR 卷积等底层工具函数。
所有函数操作 numpy 数组，不依赖 PyTorch。
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """加载任意格式音频文件，重采样到目标采样率并转为单声道。

    支持 WAV/FLAC/OGG (via soundfile) 和 M4A/MP3/AAC (via librosa fallback)。

    Args:
        file_path: 音频文件路径.
        target_sr: 目标采样率 (Hz), 默认 16000.

    Returns:
        (waveform, sample_rate): 波形 numpy 数组和实际采样率.
    """
    waveform, sr = _read_audio(file_path)
    # 多声道 → 单声道 (取均值)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = resample_if_needed(waveform, sr, target_sr)
    return waveform.astype(np.float32), target_sr


def _read_audio(file_path: str) -> tuple[np.ndarray, int]:
    """两级 fallback 读取音频：soundfile → librosa。

    soundfile 处理 WAV/FLAC/OGG (高性能)；
    librosa 处理 M4A/MP3/AAC 等压缩格式 (依赖 audioread + ffmpeg)。

    Args:
        file_path: 音频文件路径.

    Returns:
        (waveform, sample_rate).

    Raises:
        RuntimeError: 两种方式均读取失败.
    """
    # 第一级：soundfile (高速, 多数格式)
    try:
        waveform, sr = sf.read(file_path, dtype="float32")
        return waveform, sr
    except Exception:
        pass

    # 第二级：librosa (全覆盖, 依赖 ffmpeg)
    try:
        import librosa
        waveform, sr = librosa.load(file_path, sr=None, mono=False)
        return waveform.astype(np.float32), sr
    except Exception:
        raise RuntimeError(
            f"无法读取音频文件: {file_path}。"
            f"请确认文件未损坏，且已安装 ffmpeg (brew install ffmpeg 或 apt install ffmpeg)"
        )


def resample_if_needed(
    waveform: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """按需重采样音频到目标采样率。

    Args:
        waveform: 输入波形.
        orig_sr: 原始采样率 (Hz).
        target_sr: 目标采样率 (Hz).

    Returns:
        重采样后的波形，如果采样率一致则直接返回.
    """
    if orig_sr == target_sr:
        return waveform
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(waveform, up, down).astype(np.float32)


def normalize_rms(waveform: np.ndarray, target_db: float = -25.0) -> np.ndarray:
    """RMS 幅值归一化，将音频信号缩放到目标电平。

    Args:
        waveform: 输入波形.
        target_db: 目标 RMS 电平 (dBFS), 默认 -25.

    Returns:
        归一化后的波形，保持原始相对幅值关系.
    """
    rms = np.sqrt(np.mean(waveform**2) + 1e-12)
    target_rms = 10.0 ** (target_db / 20.0)
    return waveform * (target_rms / rms)


def rms_energy(waveform: np.ndarray) -> float:
    """计算信号的 RMS 能量值（线性尺度）。

    Args:
        waveform: 输入波形.

    Returns:
        RMS 能量值.
    """
    return float(np.sqrt(np.mean(waveform**2) + 1e-12))


def compute_rms_db(waveform: np.ndarray) -> float:
    """计算信号的 RMS 电平 (dBFS)。

    Args:
        waveform: 输入波形.

    Returns:
        dBFS 值.
    """
    rms = rms_energy(waveform)
    return float(20.0 * np.log10(rms + 1e-12))
