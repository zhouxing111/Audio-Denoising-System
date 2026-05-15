"""
models/audio_inpainter.py — 音频修复模块

提供三种修复策略:
  - spline:   时域三次样条插值 (短时静音/间隙)
  - spectral: 频域双线性插值 + Griffin-Lim 相位重建 (频谱局部缺失)
  - unet:     U-Net 迭代频谱修复 (通用, 需预训练 checkpoint)

同时内置 generate_damaged() 自动生成损坏音频用于评估。
"""

import logging
import random

import numpy as np
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


class AudioInpainter:
    """音频修复器。

    修复流程: detect_damage() 定位损坏段 → inpaint() 选择策略修复。
    """

    def __init__(self, n_fft: int = 512, hop_length: int = 256):
        """初始化修复器。

        Args:
            n_fft: STFT 的 FFT 点数.
            hop_length: STFT 帧移.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length

    # ================================================================
    #  损坏检测
    # ================================================================

    def detect_damage(
        self, waveform: np.ndarray, sr: int,
        silence_thresh_db: float = -40.0,
        min_silence_ms: float = 30.0,
        clip_threshold: float = 0.95,
    ) -> list[dict]:
        """检测音频中的损坏区域。

        检测两类损坏:
          - 静音段 (RMS 低于阈值)
          - 削波 (样本幅值接近 ±1.0)

        Args:
            waveform: 输入波形.
            sr: 采样率.
            silence_thresh_db: 静音判定阈值 (dBFS).
            min_silence_ms: 最小静音长度 (ms).
            clip_threshold: 削波检测阈值 (绝对幅值).

        Returns:
            损坏段列表, 每项 {"start": int, "end": int, "type": str}.
        """
        regions = []

        # 静音检测：分帧 → 逐帧 RMS → 标记连续低能量段
        frame_len = int(sr * 0.01)  # 10ms 帧
        hop = frame_len // 2
        n = len(waveform)
        n_frames = max(1, (n - frame_len) // hop + 1)
        is_silent = np.zeros(n_frames, dtype=bool)

        for i in range(n_frames):
            start = i * hop
            frame = waveform[start : start + frame_len]
            rms_db = 20.0 * np.log10(np.sqrt(np.mean(frame**2)) + 1e-12)
            is_silent[i] = rms_db < silence_thresh_db

        # 合并连续静音帧
        min_silence_frames = int(min_silence_ms / 10)
        in_silence = False
        silence_start = 0
        for i in range(n_frames):
            if is_silent[i] and not in_silence:
                in_silence = True
                silence_start = i * hop
            elif not is_silent[i] and in_silence:
                in_silence = False
                dur = i * hop - silence_start
                if dur >= min_silence_ms * sr / 1000:
                    regions.append({"start": silence_start, "end": i * hop, "type": "silence"})
        if in_silence:
            dur = n_frames * hop - silence_start
            if dur >= min_silence_ms * sr / 1000:
                regions.append({"start": silence_start, "end": n_frames * hop, "type": "silence"})

        # 削波检测
        clips = np.where(np.abs(waveform) > clip_threshold)[0]
        if len(clips) > 10:
            # 查找削波段 (连续高幅值样本)
            breaks = np.where(np.diff(clips) > sr * 0.01)[0]  # 10ms 以上间断
            seg_starts = np.concatenate([[0], breaks + 1])
            seg_ends = np.concatenate([breaks, [len(clips) - 1]])
            for s, e in zip(seg_starts, seg_ends):
                if e - s > int(sr * 0.005):  # 至少 5ms
                    regions.append({
                        "start": clips[s], "end": min(clips[e] + 1, n),
                        "type": "clipping",
                    })

        return sorted(regions, key=lambda r: r["start"])

    # ================================================================
    #  修复入口
    # ================================================================

    def inpaint(
        self, waveform: np.ndarray, sr: int,
        method: str = "spline",
        model_ckpt: str | None = None,
        n_iterations: int = 3,
    ) -> np.ndarray:
        """音频修复入口。

        Args:
            waveform: 损坏的音频波形.
            sr: 采样率.
            method: "spline" | "spectral" | "unet".
            model_ckpt: U-Net checkpoint 路径 (仅 unet 方法需要).
            n_iterations: U-Net 迭代次数.

        Returns:
            修复后的波形.
        """
        regions = self.detect_damage(waveform, sr)
        if not regions:
            logger.info("未检测到损坏区域，返回原音频")
            return waveform

        logger.info(f"检测到 {len(regions)} 个损坏区域: "
                     f"{[(r['type'], (r['end']-r['start'])/sr*1000) for r in regions]}")

        if method == "spline":
            return self._inpaint_spline(waveform, regions)
        elif method == "spectral":
            return self._inpaint_spectral(waveform, regions, sr)
        elif method == "unet":
            return self._inpaint_unet(waveform, regions, sr, model_ckpt, n_iterations)
        else:
            raise ValueError(f"未知修复方法: {method}")

    # ================================================================
    #  方法 1: 三次样条时域插值
    # ================================================================

    def _inpaint_spline(
        self, waveform: np.ndarray, regions: list[dict]
    ) -> np.ndarray:
        """用三次样条插值填充静音段。

        Args:
            waveform: 损坏波形.
            regions: 损坏区域列表.

        Returns:
            修复后波形.
        """
        repaired = waveform.copy().astype(np.float64)
        n = len(waveform)

        for r in regions:
            if r["type"] != "silence":
                continue
            a, b = r["start"], min(r["end"], n)
            if a <= 1 or b >= n - 2:
                continue  # 边界无法插值
            # 取损坏段两侧各 100 个样本作为控制点
            margin = min(100, a, n - b)
            x_ctrl = np.concatenate([
                np.arange(a - margin, a),
                np.arange(b, b + margin),
            ])
            y_ctrl = waveform[x_ctrl]
            # 对损坏段内每个样本点插值
            cs = CubicSpline(x_ctrl, y_ctrl, extrapolate=False)
            x_fill = np.arange(a, b)
            repaired[a:b] = np.nan_to_num(cs(x_fill), nan=0.0)

        return repaired.astype(np.float32)

    # ================================================================
    #  方法 2: 频域双线性插值 + Griffin-Lim
    # ================================================================

    def _inpaint_spectral(
        self, waveform: np.ndarray, regions: list[dict], sr: int,
        n_iter: int = 30,
    ) -> np.ndarray:
        """在 STFT 谱上用双线性插值填充缺失频段，Griffin-Lim 重建相位。

        Args:
            waveform: 损坏波形.
            regions: 损坏区域列表.
            sr: 采样率.
            n_iter: Griffin-Lim 迭代次数.

        Returns:
            修复后波形.
        """
        import librosa

        stft = librosa.stft(waveform.astype(np.float32), n_fft=self.n_fft,
                            hop_length=self.hop_length)
        mag = np.abs(stft)
        phase = np.angle(stft)
        n_freqs, n_frames = mag.shape

        # 将时域损坏区域映射到 STFT 帧索引
        damaged_frames = set()
        for r in regions:
            f_start = max(0, r["start"] // self.hop_length)
            f_end = min(n_frames, r["end"] // self.hop_length + 1)
            damaged_frames.update(range(f_start, f_end))

        # 对损坏帧的幅度谱做双线性插值
        from scipy.interpolate import griddata

        ok_freqs, ok_frames = [], []
        ok_vals = []
        for f in range(n_freqs):
            for t in range(n_frames):
                if t not in damaged_frames:
                    ok_freqs.append(f)
                    ok_frames.append(t)
                    ok_vals.append(mag[f, t])

        if not ok_vals:
            return waveform  # 全部损坏，无法修复

        points = np.column_stack([ok_freqs, ok_frames])
        for t in damaged_frames:
            for f in range(n_freqs):
                if np.isclose(mag[f, t], 0) or mag[f, t] < np.median(mag[:, t]) * 0.1:
                    query = np.array([[f, t]])
                    val = griddata(points, ok_vals, query, method="linear")
                    if not np.isnan(val[0]):
                        mag[f, t] = val[0]

        # Griffin-Lim 相位重建
        stft_est = mag * np.exp(1j * phase)
        for _ in range(n_iter):
            waveform_est = librosa.istft(stft_est, hop_length=self.hop_length,
                                         win_length=self.n_fft, length=len(waveform))
            stft_new = librosa.stft(waveform_est, n_fft=self.n_fft,
                                    hop_length=self.hop_length)
            stft_est = mag * np.exp(1j * np.angle(stft_new))

        return librosa.istft(stft_est, hop_length=self.hop_length,
                             win_length=self.n_fft, length=len(waveform)).astype(np.float32)

    # ================================================================
    #  方法 3: U-Net 迭代频谱修复
    # ================================================================

    def _inpaint_unet(
        self, waveform: np.ndarray, regions: list[dict], sr: int,
        model_ckpt: str | None, n_iterations: int = 3,
    ) -> np.ndarray:
        """利用预训练 U-Net 进行迭代频谱修复。

        每次迭代:
          1. 对当前 estimate 做 STFT
          2. 标记损坏区域 → 损伤区域置零
          3. U-Net 前向推理 → 预测干净频谱
          4. 用 U-Net 输出更新损坏区域 (保留正常区域不变)

        Args:
            waveform: 损坏波形.
            regions: 损坏区域列表.
            sr: 采样率.
            model_ckpt: U-Net checkpoint 路径.
            n_iterations: 迭代次数 (2~3 次即可).

        Returns:
            修复后波形.
        """
        import librosa
        import torch
        from .unet import UNetDenoiser

        if model_ckpt is None:
            raise ValueError("U-Net 修复需要 --ckpt 参数指定模型权重路径")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNetDenoiser(n_fft=self.n_fft, hop_length=self.hop_length).to(device)
        ckpt = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # 初始 estimate = 原始损坏波形
        current = waveform.copy().astype(np.float32)

        for iteration in range(n_iterations):
            stft = librosa.stft(current, n_fft=self.n_fft, hop_length=self.hop_length)
            mag = np.abs(stft)
            phase = np.angle(stft)
            n_freqs, n_frames = mag.shape

            # 标记损坏时间帧
            damaged_frames = np.zeros(n_frames, dtype=bool)
            for r in regions:
                f_start = max(0, r["start"] // self.hop_length)
                f_end = min(n_frames, r["end"] // self.hop_length + 1)
                damaged_frames[f_start:f_end] = True

            # 构建损伤掩膜 (1=正常, 0=损坏)
            mask_known = np.ones((n_freqs, n_frames), dtype=np.float32)
            mask_known[:, damaged_frames] = 0.0

            # U-Net 前向
            mag_tensor = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pred_mask = model.forward(mag_tensor).squeeze().cpu().numpy()

            # 掩蔽得到 U-Net 估计的干净幅度谱
            mag_unet = mag * pred_mask

            # 融合: 正常区域用原始, 损坏区域用 U-Net 预测
            mag_fused = mag * mask_known + mag_unet * (1.0 - mask_known)

            # iSTFT 重建
            stft_fused = mag_fused * np.exp(1j * phase)
            current = librosa.istft(stft_fused, hop_length=self.hop_length,
                                    win_length=self.n_fft, length=len(waveform))
            current = current.astype(np.float32)

            logger.info(f"U-Net 修复迭代 {iteration + 1}/{n_iterations} 完成")

        return current

    # ================================================================
    #  自动生成损坏音频 (静态方法)
    # ================================================================

    @staticmethod
    def generate_damaged(
        clean_waveform: np.ndarray,
        sr: int,
        num_silence: int = 3,
        silence_min_ms: float = 50.0,
        silence_max_ms: float = 300.0,
        clip_ratio: float = 0.3,
        seed: int | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """从纯净音频自动生成带损坏的音频。

        Args:
            clean_waveform: 纯净语音波形.
            sr: 采样率.
            num_silence: 插入的静音段数量.
            silence_min_ms: 静音段最短时长 (ms).
            silence_max_ms: 静音段最长时长 (ms).
            clip_ratio: 最大幅值的削波比例 (0.3 = 幅值 > 0.3 即削波).
            seed: 随机种子.

        Returns:
            (damaged_waveform, gt_regions): 损坏音频和真实损坏位置列表.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        damaged = clean_waveform.copy().astype(np.float32)
        n = len(damaged)
        gt_regions = []

        # 1. 插入随机静音段
        for _ in range(num_silence):
            dur = int(random.uniform(silence_min_ms, silence_max_ms) * sr / 1000)
            if dur >= n // 2:
                continue
            start = random.randint(int(sr * 0.1), max(int(sr * 0.1) + 1, n - dur - int(sr * 0.1)))
            damaged[start : start + dur] = 0.0
            gt_regions.append({"start": start, "end": start + dur, "type": "silence"})

        # 2. 施加削波
        if clip_ratio > 0:
            max_val = np.max(np.abs(damaged))
            threshold = max_val * clip_ratio
            damaged = np.clip(damaged, -threshold, threshold)

        # 3. 频率缺失 (随机掩蔽频谱区域)
        import librosa
        stft = librosa.stft(damaged, n_fft=512, hop_length=256)
        mag = np.abs(stft)
        n_freqs, n_frames = mag.shape
        if n_frames > 5:
            num_bands = random.randint(1, 3)
            for _ in range(num_bands):
                f0 = random.randint(10, n_freqs - 20)
                fw = random.randint(5, 20)
                t0 = random.randint(0, max(1, n_frames - 10))
                tw = random.randint(5, min(20, n_frames - t0))
                mag[f0 : f0 + fw, t0 : t0 + tw] = 0.0
            damaged = librosa.istft(mag * np.exp(1j * np.angle(stft)),
                                    hop_length=256, win_length=512, length=n)
            damaged = damaged.astype(np.float32)

        return damaged, sorted(gt_regions, key=lambda r: r["start"])
