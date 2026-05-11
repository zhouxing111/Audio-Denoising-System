"""
data/augment.py — 时域+频域数据增强

提供 SpecAugment 风格的频谱掩蔽和时域扰动，
在训练阶段随机增强以提升模型泛化能力。
"""

import random

import numpy as np


def spec_augment(
    mag_spec: np.ndarray,
    freq_mask_width: int = 16,
    time_mask_width: int = 20,
    freq_mask_prob: float = 0.1,
    time_mask_prob: float = 0.1,
) -> np.ndarray:
    """对幅度谱执行 SpecAugment 频率+时间掩蔽。

    在随机位置将频谱的连续频段或连续时间帧置零，
    模拟频率缺失和信号断层的退化效果。

    Args:
        mag_spec: 幅度谱, shape (n_freqs, n_frames).
        freq_mask_width: 频率掩蔽最大宽度 (频点数).
        time_mask_width: 时间掩蔽最大宽度 (帧数).
        freq_mask_prob: 频率掩蔽触发概率.
        time_mask_prob: 时间掩蔽触发概率.

    Returns:
        增强后的幅度谱 (原地修改).
    """
    n_freqs, n_frames = mag_spec.shape

    # 频率掩蔽
    if random.random() < freq_mask_prob and freq_mask_width > 0:
        w = random.randint(1, freq_mask_width)
        f0 = random.randint(0, max(0, n_freqs - w))
        mag_spec[f0 : f0 + w, :] = 0.0

    # 时间掩蔽
    if random.random() < time_mask_prob and time_mask_width > 0:
        w = random.randint(1, time_mask_width)
        t0 = random.randint(0, max(0, n_frames - w))
        mag_spec[:, t0 : t0 + w] = 0.0

    return mag_spec


def volume_perturb(
    waveform: np.ndarray, range_db: float = 3.0
) -> np.ndarray:
    """随机音量扰动，在 ±range_db 范围内均匀调整增益。

    Args:
        waveform: 输入波形.
        range_db: 音量变化范围 (dB).

    Returns:
        音量调整后的波形.
    """
    gain_db = random.uniform(-range_db, range_db)
    gain_linear = 10.0 ** (gain_db / 20.0)
    return waveform * gain_linear


def speed_perturb(
    waveform: np.ndarray, sr: int, min_rate: float = 0.9, max_rate: float = 1.1
) -> np.ndarray:
    """随机速度扰动 (通过重采样实现变速不变调效果近似)。

    注：精确的变速不变调需要 WSOLA/PSOLA 算法，
    此处采用线性重采样作为快速近似。

    Args:
        waveform: 输入波形.
        sr: 采样率 (未使用, 保留接口一致性).
        min_rate: 最低速度因子.
        max_rate: 最高速度因子.

    Returns:
        变速后波形 (长度会改变，训练时需后续裁剪/填充).
    """
    rate = random.uniform(min_rate, max_rate)
    if rate == 1.0:
        return waveform
    from scipy.signal import resample

    new_len = int(len(waveform) / rate)
    return resample(waveform, new_len).astype(np.float32)


def insert_mute_segment(
    waveform: np.ndarray, sr: int, mute_prob: float = 0.05
) -> np.ndarray:
    """随机插入静音段，模拟信号丢失。

    Args:
        waveform: 输入波形.
        sr: 采样率 (用于计算静音段长度).
        mute_prob: 插入静音段的概率.

    Returns:
        插入静音段后的波形.
    """
    if random.random() >= mute_prob:
        return waveform

    # 静音段长度 50ms~300ms
    mute_len = random.randint(int(sr * 0.05), int(sr * 0.3))
    if mute_len >= len(waveform) // 2:
        return waveform

    start = random.randint(0, len(waveform) - mute_len)
    waveform[start : start + mute_len] = 0.0
    return waveform


def apply_augmentations(
    waveform: np.ndarray,
    sr: int,
    volume_range: float = 3.0,
    mute_prob: float = 0.05,
) -> np.ndarray:
    """一键应用全部时域增强 (在混合前对纯净语音执行)。

    Args:
        waveform: 输入波形.
        sr: 采样率.
        volume_range: 音量扰动范围 (±dB).
        mute_prob: 静音段插入概率.

    Returns:
        增强后的波形.
    """
    waveform = volume_perturb(waveform, range_db=volume_range)
    waveform = insert_mute_segment(waveform, sr, mute_prob=mute_prob)
    return waveform
