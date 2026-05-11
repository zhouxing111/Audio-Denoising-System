"""
models/unet.py — U-Net 深度学习降噪模型

7层 Encoder-Decoder 架构，跳跃连接保留高频细节。
输入 STFT 幅度谱 → 预测 IRM (Ideal Ratio Mask) → 掩蔽后 iSTFT 还原波形。

训练时使用 forward() 做张量前向传播，
推理时使用 denoise_audio() 完成 STFT→掩膜→iSTFT 完整流程。
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseDenoiser


class UNetDenoiser(BaseDenoiser, nn.Module):
    """U-Net 频谱掩膜降噪模型。

    输入: 带噪音频的幅度谱 (batch, 1, n_freqs, n_frames)
    输出: 预测的 IRM 掩膜 (batch, 1, n_freqs, n_frames), 值域 [0, 1]
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        base_channels: int = 32,
    ):
        """初始化 U-Net 模型。

        Args:
            n_fft: STFT 的 FFT 点数，决定输入频谱的频率维度 (n_fft//2+1).
            hop_length: STFT 帧移.
            base_channels: 第一层卷积通道数 (每层翻倍).
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freqs = n_fft // 2 + 1  # 257 for n_fft=512

        # Encoder layers: (in_ch, out_ch)
        ch = base_channels
        self.enc1 = self._conv_block(1, ch)
        self.enc2 = self._conv_block(ch, ch * 2)
        self.enc3 = self._conv_block(ch * 2, ch * 4)
        self.enc4 = self._conv_block(ch * 4, ch * 8)
        self.enc5 = self._conv_block(ch * 8, ch * 16)
        self.enc6 = self._conv_block(ch * 16, ch * 32)
        self.enc7 = self._conv_block(ch * 32, ch * 64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._conv_block(ch * 64, ch * 128)

        # Decoder layers
        self.up7 = nn.ConvTranspose2d(
            ch * 128, ch * 64, kernel_size=2, stride=2
        )
        self.dec7 = self._conv_block(ch * 128, ch * 64)

        self.up6 = nn.ConvTranspose2d(
            ch * 64, ch * 32, kernel_size=2, stride=2
        )
        self.dec6 = self._conv_block(ch * 64, ch * 32)

        self.up5 = nn.ConvTranspose2d(
            ch * 32, ch * 16, kernel_size=2, stride=2
        )
        self.dec5 = self._conv_block(ch * 32, ch * 16)

        self.up4 = nn.ConvTranspose2d(
            ch * 16, ch * 8, kernel_size=2, stride=2
        )
        self.dec4 = self._conv_block(ch * 16, ch * 8)

        self.up3 = nn.ConvTranspose2d(
            ch * 8, ch * 4, kernel_size=2, stride=2
        )
        self.dec3 = self._conv_block(ch * 8, ch * 4)

        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(ch * 4, ch * 2)

        self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(ch * 2, ch)

        # Output layer: 1×1 conv → sigmoid
        self.out_conv = nn.Conv2d(ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """创建双卷积块 (Conv2d → BN → ReLU) × 2。

        Args:
            in_ch: 输入通道数.
            out_ch: 输出通道数.

        Returns:
            顺序卷积块.
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _pad_to_match(self, x: Tensor) -> tuple[Tensor, int, int]:
        """沿时间维填充使尺寸适合下采样 (需要能被 2^6=64 整除)。

        Args:
            x: 输入张量 (batch, 1, n_freqs, n_frames).

        Returns:
            (padded_x, orig_frames, padded_frames).
        """
        _, _, n_freqs, n_frames = x.shape
        divisor = 64
        if n_frames % divisor == 0:
            return x, n_frames, n_frames
        pad_target = ((n_frames // divisor) + 1) * divisor
        pad_amount = pad_target - n_frames
        return torch.nn.functional.pad(x, (0, pad_amount)), n_frames, pad_target

    def forward(self, x: Tensor) -> Tensor:
        """前向传播，预测 IRM 掩膜。

        Args:
            x: 带噪幅度谱, shape (batch, 1, n_freqs, n_frames).

        Returns:
            预测掩膜, shape (batch, 1, n_freqs, n_frames_padded).
        """
        x, orig_frames, _ = self._pad_to_match(x)

        # Encoder (with skip connections)
        e1 = self.enc1(x)       # (B, ch,   F, T)
        e2 = self.enc2(self.pool(e1))  # (B, ch*2,  F/2, T/2)
        e3 = self.enc3(self.pool(e2))  # (B, ch*4,  F/4, T/4)
        e4 = self.enc4(self.pool(e3))  # (B, ch*8,  F/8, T/8)
        e5 = self.enc5(self.pool(e4))  # (B, ch*16, F/16, T/16)
        e6 = self.enc6(self.pool(e5))  # (B, ch*32, F/32, T/32)
        e7 = self.enc7(self.pool(e6))  # (B, ch*64, F/64, T/64)

        # Bottleneck
        b = self.bottleneck(self.pool(e7))  # (B, ch*128, F/128, T/128)

        # Decoder (with skip connections)
        d7 = self.up7(b)
        d7 = self._align_and_concat(d7, e7)
        d7 = self.dec7(d7)

        d6 = self.up6(d7)
        d6 = self._align_and_concat(d6, e6)
        d6 = self.dec6(d6)

        d5 = self.up5(d6)
        d5 = self._align_and_concat(d5, e5)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = self._align_and_concat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._align_and_concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._align_and_concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._align_and_concat(d1, e1)
        d1 = self.dec1(d1)

        mask = self.sigmoid(self.out_conv(d1))
        # 裁剪回原始时间帧数
        if mask.shape[-1] > orig_frames:
            mask = mask[..., :orig_frames]
        return mask

    @staticmethod
    def _align_and_concat(dec: Tensor, enc: Tensor) -> Tensor:
        """对齐 decoder 与 encoder 空间尺寸后沿通道维拼接。

        处理奇数尺寸导致的 ±1 像素偏差 (如 257 → 256):
        - decoder 偏大时裁剪
        - decoder 偏小时零填充

        Args:
            dec: decoder 上采样特征.
            enc: encoder 跳跃连接特征.

        Returns:
            拼接后的特征.
        """
        _, _, h_enc, w_enc = enc.shape
        _, _, h_dec, w_dec = dec.shape

        # 频率维对齐
        if h_dec > h_enc:
            dec = dec[:, :, :h_enc, :]
        elif h_dec < h_enc:
            dec = torch.nn.functional.pad(dec, (0, 0, 0, h_enc - h_dec))

        # 时间维对齐
        if w_dec > w_enc:
            dec = dec[:, :, :, :w_enc]
        elif w_dec < w_enc:
            dec = torch.nn.functional.pad(dec, (0, w_enc - w_dec))

        return torch.cat([dec, enc], dim=1)

    def denoise_audio(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """端到端降噪推理。

        流程: 音频 → STFT → 幅度谱 → 模型预测掩膜 → 掩蔽 → iSTFT → 音频

        Args:
            waveform: 带噪音频, shape (n_samples,).
            sr: 采样率 (Hz).

        Returns:
            降噪后音频, shape (n_samples,).
        """
        self.eval()
        device = next(self.parameters()).device

        # STFT
        stft = self._compute_stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length
        )
        mag = np.abs(stft)
        phase = np.angle(stft)

        # 转 Tensor
        mag_tensor = (
            torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float().to(device)
        )  # (1, 1, F, T)

        with torch.no_grad():
            mask = self.forward(mag_tensor)  # (1, 1, F, T)
            mask = mask.squeeze().cpu().numpy()  # (F, T)

        # 掩蔽 + iSTFT
        denoised_mag = mag * mask
        denoised_stft = denoised_mag * np.exp(1j * phase)
        denoised_waveform = self._compute_istft(
            denoised_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            length=len(waveform),
        )
        return denoised_waveform.astype(np.float32)
