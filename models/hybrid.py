"""
models/hybrid.py — HybridDenoiser: U-Net 动态噪声估计 + Wiener 保守降噪

将 U-Net 的 IRM 掩膜作为动态噪声 PSD 估计器，驱动维纳滤波。
在分布外数据上自动切换到保守模式，避免过度切除。
完全不修改 U-Net 权重，只改变推理逻辑。

原理:
  1. U-Net 预测 mask ∈ [0,1]
  2. 噪声 PSD: noise_psd = (1 - mask)² × |Y|²
  3. Wiener gain: G = |Y|² / (|Y|² + noise_psd)
  4. 自适应融合: 高置信度区域信任 U-Net, 低置信度区域信任 Wiener
"""

import numpy as np
import torch

from .unet import UNetDenoiser


class HybridDenoiser:
    """U-Net + Wiener 混合降噪器。

    U-Net 定位"噪声在哪里"，Wiener 决定"降多少"。
    对训练集见过的数据 ≈ 原版 U-Net，对分布外数据自动保守降噪。
    """

    def __init__(self, n_fft: int = 512, hop_length: int = 256):
        """初始化混合降噪器。

        Args:
            n_fft: STFT FFT 点数.
            hop_length: STFT 帧移.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._model = None
        self._device = None

    def _load_model(self, ckpt_path: str) -> None:
        """加载 U-Net 模型（仅首次调用时）。

        Args:
            ckpt_path: checkpoint 路径.
        """
        if self._model is not None:
            return
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = UNetDenoiser(n_fft=self.n_fft, hop_length=self.hop_length).to(self._device)
        ckpt = torch.load(ckpt_path, map_location=self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

    def denoise_audio(
        self, waveform: np.ndarray, sr: int,
        model_ckpt: str | None = None, wiener_strength: float = 1.0,
    ) -> np.ndarray:
        """混合降噪。

        Args:
            waveform: 带噪音频波形.
            sr: 采样率 (Hz).
            model_ckpt: U-Net checkpoint 路径.
            wiener_strength: Wiener 强度系数 (>1 = 更保守, <1 = 更激进).

        Returns:
            降噪后波形.
        """
        if model_ckpt is None:
            raise ValueError("Hybrid 需要 --ckpt 参数指定 U-Net 权重路径")
        self._load_model(model_ckpt)

        import librosa

        # 1. STFT
        stft = librosa.stft(waveform.astype(np.float32), n_fft=self.n_fft,
                            hop_length=self.hop_length, win_length=self.n_fft)
        mag = np.abs(stft)
        phase = np.angle(stft)

        # 2. U-Net 预测掩膜
        mag_tensor = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float().to(self._device)
        with torch.no_grad():
            mask = self._model.forward(mag_tensor).squeeze().cpu().numpy()

        # 3. 掩膜后处理 (与 U-Net 一致: 平滑 + 地板 + 压缩)
        from scipy.ndimage import median_filter, uniform_filter1d
        mask = uniform_filter1d(mask.astype(np.float64), size=5, axis=1)
        mask = median_filter(mask, size=(3, 1))
        mask = np.maximum(mask, 0.05)
        mask = np.sqrt(mask).astype(np.float32)

        # 4. 动态噪声 PSD: 从 U-Net 掩膜推算
        noise_mask = 1.0 - mask   # U-Net 认为"噪声"的区域
        noise_psd = (noise_mask ** 2) * (mag ** 2)

        # 5. Wiener 增益 (用 U-Net 估计的动态噪声 PSD)
        wiener_gain = (mag ** 2) / (mag ** 2 + wiener_strength * noise_psd + 1e-8)
        wiener_gain = np.clip(wiener_gain, 0.0, 1.0)

        # 6. 自适应融合: U-Net 置信度 = mask
        #    高置信度 → 倾向 U-Net; 低置信度 → 倾向 Wiener
        final_gain = mask * mask + (1.0 - mask) * wiener_gain
        final_gain = np.clip(final_gain, 0.03, 1.0)

        # 7. 掩蔽 + 10% 原始信号混合
        denoised_mag = mag * final_gain
        denoised_mag = 0.88 * denoised_mag + 0.12 * mag

        # 8. iSTFT
        denoised_stft = denoised_mag * np.exp(1j * phase)
        waveform_out = librosa.istft(denoised_stft, hop_length=self.hop_length,
                                     win_length=self.n_fft, length=len(waveform))
        waveform_out = waveform_out.astype(np.float32)

        # 9. 边界淡出
        fade_len = min(256, len(waveform_out) // 4)
        if fade_len > 0:
            fade = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_len) / fade_len)
            waveform_out[:fade_len] *= fade.astype(np.float32)
            waveform_out[-fade_len:] *= fade[::-1].astype(np.float32)

        return waveform_out
