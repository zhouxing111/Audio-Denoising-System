"""
data/noise_diagnosis.py — 噪声类型自动诊断

基于 VAD (Voice Activity Detection) 分离语音帧和非语音帧，
分析非语音帧的频谱特征，自动分类噪声类型:
- white_noise: 全频带平坦分布
- low_freq_hum: 低频集中 (<500Hz, 典型的电器嗡嗡声)
- background_speech: 中频波动 (300Hz~3kHz, 人声 babble)
- high_freq_noise: 高频集中 (>4kHz, 电子噪声)
"""

import numpy as np

# 噪声类型标签
NOISE_TYPES = {
    "white_noise": "白噪声 (全频带平坦)",
    "low_freq_hum": "低频嗡嗡声 (电器/工频噪声)",
    "background_speech": "背景人声 (Babble/多人交谈)",
    "high_freq_noise": "高频电子噪声",
    "mixed_noise": "混合噪声",
}


def detect_speech_frames(
    waveform: np.ndarray,
    sr: int,
    frame_ms: float = 30.0,
    energy_threshold: float = 0.05,
) -> np.ndarray:
    """基于短时能量和过零率检测语音帧。

    返回布尔数组，True 表示语音帧，False 表示非语音/噪声帧。

    Args:
        waveform: 输入波形.
        sr: 采样率 (Hz).
        frame_ms: 帧长 (ms).
        energy_threshold: 相对能量阈值 (0~1), 低于此值的帧视为静音/噪声.

    Returns:
        is_speech: 布尔数组, shape (n_frames,).
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop = frame_len // 2
    n = len(waveform)
    n_frames = max(1, (n - frame_len) // hop + 1)

    # 全局能量归一化阈值
    global_max_energy = 0.0
    energies = np.zeros(n_frames)
    zcrs = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        frame = waveform[start : start + frame_len].astype(np.float64)
        energies[i] = np.mean(frame**2)
        global_max_energy = max(global_max_energy, energies[i])
        # 过零率
        zcrs[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_len)

    # 归一化能量
    if global_max_energy > 1e-12:
        energies = energies / global_max_energy

    # 高能量 + 中等过零率 → 语音帧
    zcr_mean = np.mean(zcrs)
    is_speech = (energies > energy_threshold) & (zcrs > zcr_mean * 0.3)
    return is_speech


def analyze_noise_spectrum(
    waveform: np.ndarray,
    sr: int,
    n_fft: int = 512,
    is_speech: np.ndarray | None = None,
    frame_ms: float = 30.0,
) -> dict:
    """分析非语音帧的频谱特征，返回频段能量分布。

    Args:
        waveform: 输入波形.
        sr: 采样率 (Hz).
        n_fft: FFT 点数.
        is_speech: 语音帧掩码 (None 则自动检测).
        frame_ms: 分帧时长 (ms).

    Returns:
        spectrum_profile: 包含 freq_bins, noise_spectrum, band_energies 的字典.
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop = frame_len // 2
    n = len(waveform)
    n_frames = max(1, (n - frame_len) // hop + 1)

    if is_speech is None:
        is_speech = detect_speech_frames(waveform, sr, frame_ms)

    # 收集非语音帧的幅度谱
    noise_mags = []
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    for i in range(n_frames):
        if is_speech[i]:
            continue
        start = i * hop
        frame = waveform[start : start + frame_len].astype(np.float64)
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spec = np.abs(np.fft.rfft(frame * np.hanning(n_fft), n=n_fft))
        noise_mags.append(spec)

    if not noise_mags:
        # 全是语音帧，降级使用全部帧
        for i in range(min(n_frames, 50)):
            start = i * hop
            frame = waveform[start : start + frame_len].astype(np.float64)
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            spec = np.abs(np.fft.rfft(frame * np.hanning(n_fft), n=n_fft))
            noise_mags.append(spec)

    noise_mags = np.array(noise_mags)  # (n_noise_frames, n_freqs)
    mean_noise_spec = np.mean(noise_mags, axis=0)
    # 归一化
    max_val = np.max(mean_noise_spec)
    if max_val > 0:
        mean_noise_spec = mean_noise_spec / max_val

    # 频段划分
    band_edges = {
        "low": (0, 500),
        "mid": (500, 3000),
        "high": (3000, sr // 2),
    }
    band_energies = {}
    for band, (fl, fh) in band_edges.items():
        mask = (freqs >= fl) & (freqs < fh)
        band_energies[band] = float(np.sum(mean_noise_spec[mask]))

    # 低频特殊检测: 50Hz 工频 + 谐波
    harmonic_energy = 0.0
    for hz in [50, 100, 150, 200, 250, 300]:
        idx = np.argmin(np.abs(freqs - hz))
        harmonic_energy += float(mean_noise_spec[idx])
    band_energies["harmonic"] = harmonic_energy

    # 平坦度: 频谱方差越小越平坦 (白噪声特征)
    spectral_flatness = float(np.var(mean_noise_spec))

    return {
        "freqs": freqs,
        "noise_spectrum": mean_noise_spec,
        "band_energies": band_energies,
        "spectral_flatness": spectral_flatness,
        "dominant_freq": float(freqs[np.argmax(mean_noise_spec)]),
    }


def classify_noise_type(spectrum_profile: dict) -> tuple[str, str, dict]:
    """根据频谱特征分类噪声类型。

    分类逻辑:
    - 频谱平坦 (低方差) → 白噪声
    - 低频占主导 + 工频谐波能量高 → 低频嗡嗡声
    - 中频占主导 → 背景人声
    - 高频占主导 → 高频电子噪声
    - 否则 → 混合噪声

    Args:
        spectrum_profile: analyze_noise_spectrum 返回的频谱特征字典.

    Returns:
        (noise_type_key, noise_type_label, diagnosis_details): 噪声分类结果.
    """
    band = spectrum_profile["band_energies"]
    flatness = spectrum_profile["spectral_flatness"]
    total = band.get("low", 0) + band.get("mid", 0) + band.get("high", 0) + 1e-12

    details = {
        "low_ratio": round(band.get("low", 0) / total, 4),
        "mid_ratio": round(band.get("mid", 0) / total, 4),
        "high_ratio": round(band.get("high", 0) / total, 4),
        "harmonic_strength": round(band.get("harmonic", 0), 4),
        "spectral_flatness": round(flatness, 4),
        "dominant_freq_hz": round(spectrum_profile["dominant_freq"], 1),
    }

    # 判断逻辑
    if flatness < 0.008 and max(
        band.get("low", 0), band.get("mid", 0), band.get("high", 0)
    ) < 1.5 * min(
        band.get("low", 0) + 0.01,
        band.get("mid", 0) + 0.01,
    ):
        return "white_noise", NOISE_TYPES["white_noise"], details

    if band.get("harmonic", 0) > 0.3 and details["low_ratio"] > 0.4:
        return "low_freq_hum", NOISE_TYPES["low_freq_hum"], details

    if details["mid_ratio"] > 0.4:
        return "background_speech", NOISE_TYPES["background_speech"], details

    if details["high_ratio"] > 0.35:
        return "high_freq_noise", NOISE_TYPES["high_freq_noise"], details

    return "mixed_noise", NOISE_TYPES["mixed_noise"], details


def diagnose_noise(
    waveform: np.ndarray, sr: int
) -> tuple[str, str, dict, dict]:
    """一键噪声诊断：检测语音帧 → 频谱分析 → 噪声分类。

    这是供 GUI 调用的主入口函数。

    Args:
        waveform: 输入带噪音频波形.
        sr: 采样率 (Hz).

    Returns:
        (noise_type_key, noise_type_label, details_dict, spectrum_profile):
            噪声类型标识、中文标签、详细占比、完整频谱数据.
    """
    is_speech = detect_speech_frames(waveform, sr)
    n_noise = int(np.sum(~is_speech))
    n_total = len(is_speech)

    profile = analyze_noise_spectrum(waveform, sr, is_speech=is_speech)
    noise_type, noise_label, details = classify_noise_type(profile)

    # 附加帧统计
    details["noise_frame_ratio"] = round(n_noise / max(n_total, 1), 4)
    details["noise_frames"] = n_noise
    details["total_frames"] = n_total

    return noise_type, noise_label, details, profile
