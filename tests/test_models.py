"""
tests/test_models.py — 模型模块单元测试

验证传统算法和 U-Net 模型的前向传播和推理接口。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_wiener_output_shape():
    """维纳滤波输出应与输入等长。"""
    import numpy as np

    from models.wiener import WienerFilter

    sr = 16000
    waveform = np.random.randn(sr * 2).astype(np.float32)
    wf = WienerFilter()
    denoised = wf.denoise_audio(waveform, sr)
    assert len(denoised) == len(waveform), (
        f"输出长度 {len(denoised)} != 输入长度 {len(waveform)}"
    )
    assert denoised.dtype == np.float32


def test_spectral_sub_output_shape():
    """谱减法输出应与输入等长。"""
    import numpy as np

    from models.spectral_sub import SpectralSubtraction

    sr = 16000
    waveform = np.random.randn(sr * 2).astype(np.float32)
    ss = SpectralSubtraction()
    denoised = ss.denoise_audio(waveform, sr)
    assert len(denoised) == len(waveform), (
        f"输出长度 {len(denoised)} != 输入长度 {len(waveform)}"
    )
    assert denoised.dtype == np.float32


def test_unet_forward_shape():
    """U-Net forward 输出掩膜 shape 应与输入幅度谱匹配。"""
    import torch
    from models.unet import UNetDenoiser

    model = UNetDenoiser(n_fft=512, hop_length=256)
    # 输入: (batch=2, 1, n_freqs=257, n_frames=100)
    x = torch.randn(2, 1, 257, 100)
    mask = model.forward(x)
    assert mask.shape[0] == 2
    assert mask.shape[1] == 1
    assert mask.shape[2] == 257
    # 帧数可能因 padding 略有不同
    assert torch.all((mask >= 0) & (mask <= 1)), "掩膜值应在 [0, 1] 范围内"


def test_unet_denoise_audio():
    """U-Net denoise_audio 输出应与输入等长。"""
    import numpy as np
    import torch
    from models.unet import UNetDenoiser

    model = UNetDenoiser(n_fft=512, hop_length=256)
    model.eval()
    sr = 16000
    waveform = np.random.randn(sr * 2).astype(np.float32)
    denoised = model.denoise_audio(waveform, sr)
    assert len(denoised) == len(waveform), (
        f"输出长度 {len(denoised)} != 输入长度 {len(waveform)}"
    )
    assert denoised.dtype == np.float32
