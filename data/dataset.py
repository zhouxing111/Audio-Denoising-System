"""
data/dataset.py — 动态在线混合数据集

基于 PyTorch Dataset，在 __getitem__ 阶段实时合成带噪语音。
随机截取纯净语音与噪声片段，按随机 SNR 混合，返回 (noisy, clean) 对。
按 speaker_id 独立分割 train/val/test，防止说话人泄露。
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import normalize_rms


class DenoisingDataset(Dataset):
    """动态在线语音增强数据集。

    每次 __getitem__ 调用时：
    1. 随机选择纯净语音和噪声文件
    2. 截取等长片段
    3. 按随机 SNR 在线混合
    4. 可选数据增强
    """

    def __init__(
        self,
        clean_dir: str,
        noise_dir: str,
        sample_rate: int = 16000,
        duration: float = 4.0,
        snr_low: float = -5.0,
        snr_high: float = 15.0,
        target_rms_db: float = -25.0,
        speaker_ids: list[str] | None = None,
    ):
        """初始化数据集。

        Args:
            clean_dir: 纯净语音文件目录.
            noise_dir: 噪声文件目录.
            sample_rate: 统一采样率 (Hz).
            duration: 每次截取的片段长度 (秒).
            snr_low: 随机 SNR 下限 (dB).
            snr_high: 随机 SNR 上限 (dB).
            target_rms_db: 目标 RMS 归一化电平 (dBFS).
            speaker_ids: 限定使用的说话人 ID 列表 (用于 train/val/test 分割).
                         None 表示使用全部文件.
        """
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.target_rms_db = target_rms_db

        self.clean_files = self._scan_clean(clean_dir, speaker_ids)
        self.noise_files = self._scan_noise(noise_dir)

        assert len(self.clean_files) > 0, f"未在 {clean_dir} 中找到纯净语音文件"
        assert len(self.noise_files) > 0, f"未在 {noise_dir} 中找到噪声文件"

    def _scan_clean(
        self, root: str, speaker_ids: list[str] | None
    ) -> list[Path]:
        """扫描纯净语音目录，按 speaker_id 过滤。

        Args:
            root: 纯净语音根目录.
            speaker_ids: 允许的 speaker ID 列表.

        Returns:
            符合条件的 WAV 文件路径列表.
        """
        root = Path(root)
        wavs = sorted(root.rglob("*.wav"))
        if speaker_ids is None:
            return wavs
        return [w for w in wavs if any(sid in str(w) for sid in speaker_ids)]

    def _scan_noise(self, root: str) -> list[Path]:
        """扫描噪声目录，收集所有 WAV 文件。

        Args:
            root: 噪声根目录.

        Returns:
            WAV 文件路径列表.
        """
        return sorted(Path(root).rglob("*.wav"))

    def __len__(self) -> int:
        """返回数据集长度 (固定 10000 步为一个虚拟 epoch)。"""
        return 10000

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取一对 (noisy, clean) 训练样本。

        Args:
            idx: 索引 (动态混合中忽略).

        Returns:
            (noisy_waveform, clean_waveform): shape 均为 (num_samples,).
        """
        # 1. 随机选取并加载纯净语音片段
        clean = self._load_random_segment(self.clean_files)
        # 2. 随机选取并加载噪声片段
        noise = self._load_random_segment(self.noise_files)
        # 3. 按随机 SNR 混合
        snr = random.uniform(self.snr_low, self.snr_high)
        noisy = self._mix_at_snr(clean, noise, snr)
        # 4. RMS 归一化 (对带噪语音)
        noisy = normalize_rms(noisy, self.target_rms_db)
        clean = normalize_rms(clean, self.target_rms_db)
        return torch.from_numpy(noisy), torch.from_numpy(clean)

    def _load_random_segment(self, file_list: list[Path]) -> np.ndarray:
        """随机选择一个文件并截取等长片段。

        使用统一的 load_audio() 加载，支持 WAV/FLAC/M4A/MP3 等格式。

        Args:
            file_list: 音频文件路径列表.

        Returns:
            截取的波形片段, shape (num_samples,).
        """
        from .preprocess import load_audio

        fpath = random.choice(file_list)
        waveform, _ = load_audio(str(fpath), target_sr=self.sample_rate)
        # 如果音频短于所需长度，循环填充
        if len(waveform) < self.num_samples:
            waveform = _pad_loop(waveform, self.num_samples)
        # 随机起始位置截取
        start = random.randint(0, len(waveform) - self.num_samples)
        return waveform[start : start + self.num_samples].astype(np.float32)

    def _mix_at_snr(
        self, clean: np.ndarray, noise: np.ndarray, snr_db: float
    ) -> np.ndarray:
        """按指定 SNR 将纯净语音与噪声混合。

        Args:
            clean: 纯净语音波形.
            noise: 噪声波形 (需与 clean 等长).
            snr_db: 目标信噪比 (dB).

        Returns:
            混合后的带噪语音.
        """
        clean_rms = np.sqrt(np.mean(clean**2) + 1e-12)
        noise_rms = np.sqrt(np.mean(noise**2) + 1e-12)
        target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
        noise = noise * (target_noise_rms / (noise_rms + 1e-12))
        return clean + noise


class PremixedDataset(Dataset):
    """预混合数据集 — __getitem__ 直接加载 .npy 文件，极快。

    必须先运行 scripts/premix_dataset.py 生成预混合数据。
    与 DenoisingDataset 不同，此 Dataset 不做任何在线处理。
    """

    def __init__(self, premix_dir: str):
        """初始化预混合数据集。

        Args:
            premix_dir: 包含 noisy_XXXXXX.npy 和 clean_XXXXXX.npy 的目录.
        """
        self.dir = Path(premix_dir)
        self.noisy_files = sorted(self.dir.glob("noisy_*.npy"))
        self.clean_files = sorted(self.dir.glob("clean_*.npy"))
        assert len(self.noisy_files) > 0, f"未在 {premix_dir} 找到 noisy_*.npy 文件"
        assert len(self.noisy_files) == len(self.clean_files), (
            f"noisy({len(self.noisy_files)}) 与 clean({len(self.clean_files)}) 数量不匹配"
        )

    def __len__(self) -> int:
        """返回数据集大小。"""
        return len(self.noisy_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """加载第 idx 对预混合样本。

        Args:
            idx: 样本索引.

        Returns:
            (noisy_tensor, clean_tensor): shape 均为 (num_samples,).
        """
        noisy = np.load(self.noisy_files[idx]).astype(np.float32)
        clean = np.load(self.clean_files[idx]).astype(np.float32)
        return torch.from_numpy(noisy), torch.from_numpy(clean)


def _pad_loop(waveform: np.ndarray, target_len: int) -> np.ndarray:
    """循环填充短音频到目标长度。

    Args:
        waveform: 输入波形.
        target_len: 目标采样点数.

    Returns:
        填充后的波形.
    """
    repeats = target_len // len(waveform) + 1
    return np.tile(waveform, repeats)[:target_len]
