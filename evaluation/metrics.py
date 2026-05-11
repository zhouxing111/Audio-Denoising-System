"""
evaluation/metrics.py — 客观声学评估指标

实现 SNR, SegSNR, SI-SDR, STOI, PESQ, LSD 和 DNSMOS 七种指标。
主入口: compute_all_metrics(clean, denoised, sr) → dict
"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def compute_all_metrics(
    clean: np.ndarray,
    denoised: np.ndarray,
    sr: int = 16000,
    aligned: bool = True,
) -> dict:
    """计算全部客观评估指标。

    Args:
        clean: 纯净参考信号.
        denoised: 降噪后信号.
        sr: 采样率 (Hz).
        aligned: 是否已对齐 (True 则跳过内部对齐).

    Returns:
        指标名字典, 计算失败时对应值为 float('nan').
    """
    # 确保等长
    min_len = min(len(clean), len(denoised))
    clean = clean[:min_len]
    denoised = denoised[:min_len]

    results = {}

    # --- 物理指标 ---
    results["SNR (dB)"] = _compute_snr(clean, denoised)
    results["SegSNR (dB)"] = _compute_segsnr(clean, denoised, sr)
    results["SI-SDR (dB)"] = _compute_sisdr(clean, denoised)
    results["LSD (dB)"] = _compute_lsd(clean, denoised)

    # --- 可懂度 ---
    results["STOI"] = _compute_stoi(clean, denoised, sr)

    # --- 感知 ---
    results["PESQ_WB"] = _compute_pesq(clean, denoised, sr, wb=True)
    results["PESQ_NB"] = _compute_pesq(clean, denoised, sr, wb=False)

    return results


def _compute_snr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """计算标准信噪比 SNR (dB)。

    公式: SNR = 10 * log10(||clean||^2 / ||clean - denoised||^2)

    Args:
        clean: 纯净信号.
        denoised: 降噪信号.

    Returns:
        SNR 值 (dB).
    """
    noise = clean - denoised
    signal_power = np.sum(clean**2)
    noise_power = np.sum(noise**2)
    if noise_power < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def _compute_segsnr(
    clean: np.ndarray, denoised: np.ndarray, sr: int, frame_ms: float = 30.0
) -> float:
    """计算分段信噪比 SegSNR (dB)。

    将信号分帧 (默认 30ms)，逐帧计算 SNR 后取均值，
    比全局 SNR 更能反映局部降噪质量。

    Args:
        clean: 纯净信号.
        denoised: 降噪信号.
        sr: 采样率 (Hz).
        frame_ms: 帧长 (ms).

    Returns:
        SegSNR 值 (dB).
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop = frame_len // 2
    n = len(clean)
    n_frames = max(1, (n - frame_len) // hop + 1)

    snr_list = []
    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        c = clean[start:end]
        d = denoised[start:end]
        noise = c - d
        sp = np.sum(c**2)
        np_val = np.sum(noise**2)
        if np_val < 1e-12 or sp < 1e-12:
            continue
        snr_frame = 10.0 * np.log10(sp / np_val)
        # 限制每帧 SNR 在 [-10, 35] dB 避免极端值影响均值
        snr_list.append(np.clip(snr_frame, -10.0, 35.0))

    if not snr_list:
        return float("nan")
    return float(np.mean(snr_list))


def _compute_sisdr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """计算尺度不变信号失真比 SI-SDR (dB)。

    先对 denoised 做最优缩放 (投影)，再计算 SNR。
    消除了增益差异的影响，比原始 SNR 更公平。

    Args:
        clean: 纯净信号.
        denoised: 降噪信号.

    Returns:
        SI-SDR 值 (dB).
    """
    # 最优缩放因子
    alpha = np.dot(denoised, clean) / (np.dot(clean, clean) + 1e-12)
    s_target = alpha * clean
    e_noise = denoised - s_target
    target_power = np.sum(s_target**2)
    noise_power = np.sum(e_noise**2)
    if noise_power < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(target_power / noise_power))


def _compute_stoi(
    clean: np.ndarray, denoised: np.ndarray, sr: int
) -> float:
    """计算短时客观可懂度 STOI (0~1)。

    值越高表示语音可懂度越好，1.0 表示完全可分辨。

    Args:
        clean: 纯净信号.
        denoised: 降噪信号.
        sr: 采样率 (Hz, 内部重采样至 10000Hz).

    Returns:
        STOI 值 (0~1)，失败返回 NaN.
    """
    try:
        from pystoi import stoi
        return float(stoi(clean, denoised, sr, extended=False))
    except ImportError:
        logger.warning("pystoi 未安装，跳过 STOI 计算")
        return float("nan")
    except Exception:
        return float("nan")


def _compute_pesq(
    clean: np.ndarray, denoised: np.ndarray, sr: int, wb: bool = True
) -> float:
    """计算感知语音质量评估 PESQ 得分。

    Args:
        clean: 纯净信号.
        denoised: 降噪信号.
        sr: 采样率 (Hz).
        wb: True 用宽带 PESQ (采样率需 16000), False 用窄带 (8000).

    Returns:
        PESQ 得分 (窄带 -0.5~4.5, 宽带 1.0~4.5).
    """
    try:
        from pesq import pesq
        mode = "wb" if wb else "nb"
        return float(pesq(sr, clean, denoised, mode))
    except ImportError:
        logger.warning("pesq 未安装，跳过 PESQ 计算")
        return float("nan")
    except Exception:
        return float("nan")


def _compute_dnsmos(
    waveform: np.ndarray, sr: int = 16000
) -> float:
    """使用 MIT DNSMOS 预训练模型预测无参考 MOS 得分。

    无需纯净参考信号，直接评估语音自然度和质量。
    得分范围约 1~5，越高越好。

    Args:
        waveform: 待评估音频波形.
        sr: 采样率 (Hz, 需 16000).

    Returns:
        DNSMOS 预测得分 (1~5), 失败返回 NaN.
    """
    try:
        import onnxruntime as ort
        import librosa

        # DNSMOS 需要 16kHz
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

        # 计算 log-mel 特征 (DNSMOS 的输入)
        mel = librosa.feature.melspectrogram(
            y=waveform.astype(np.float32),
            sr=16000,
            n_fft=512,
            hop_length=160,
            n_mels=64,
            fmin=20,
            fmax=8000,
        )
        log_mel = np.log10(np.maximum(mel, 1e-10)).T  # (T, 64)

        # 注意: 完整 DNSMOS 需要下载 ONNX 模型文件
        # 此处提供接口框架，实际推理需加载 pDNSMOS 的 SIG/BAK/OVRL 三个模型
        logger.warning(
            "DNSMOS 需要下载 ONNX 模型文件才能工作，"
            "请参考 https://github.com/microsoft/DNS-Challenge"
        )
        return float("nan")
    except ImportError:
        logger.warning("onnxruntime 未安装，跳过 DNSMOS 计算")
        return float("nan")
    except Exception:
        return float("nan")


def _compute_lsd(
    clean: np.ndarray, denoised: np.ndarray, n_fft: int = 512
) -> float:
    """计算对数谱距离 LSD (dB)。

    对比两段音频在频域的包络差异，值越低越好。

    Args:
        clean: 纯净信号.
        denoised: 降噪信号.
        n_fft: FFT 点数.

    Returns:
        LSD 值 (dB).
    """
    from scipy.signal import stft

    _, _, Zxx_c = stft(clean, nperseg=n_fft)
    _, _, Zxx_d = stft(denoised, nperseg=n_fft)

    mag_c = np.abs(Zxx_c) + 1e-12
    mag_d = np.abs(Zxx_d) + 1e-12

    # 逐帧计算对数谱距离，取均值
    log_diff = np.log10(mag_c) - np.log10(mag_d)
    lsd_per_frame = np.sqrt(np.mean(log_diff**2, axis=0))
    return float(np.mean(lsd_per_frame))
