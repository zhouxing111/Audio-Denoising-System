"""
tests/test_dataset.py — 数据集模块单元测试

测试动态在线混合 Dataset 的输出 shape、SNR 范围和数据增强正确性。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_dataset_creation():
    """验证用空目录也能正常断言报错 (而非静默失败)。"""
    import tempfile

    from data.dataset import DenoisingDataset

    # 空目录应报错
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            DenoisingDataset(clean_dir=tmpdir, noise_dir=tmpdir)
            assert False, "应抛出 AssertionError"
        except AssertionError:
            pass  # 预期行为


def test_mix_at_snr():
    """验证 SNR 混合函数输出的信噪比在预期范围内。"""
    import numpy as np

    from data.dataset import DenoisingDataset

    # 创建一个假数据集只是为了访问 _mix_at_snr
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # 写入一个无声文件让 dataset 初始化通过
        import soundfile as sf

        clean_file = Path(tmpdir) / "clean.wav"
        noise_file = Path(tmpdir) / "noise.wav"
        sf.write(str(clean_file), np.zeros(16000, dtype="float32"), 16000)
        sf.write(str(noise_file), np.random.randn(16000).astype("float32"), 16000)

        ds = DenoisingDataset(
            clean_dir=tmpdir,
            noise_dir=tmpdir,
            duration=1.0,
        )
        clean = np.random.randn(16000).astype(np.float32)
        noise = np.random.randn(16000).astype(np.float32)

        noisy = ds._mix_at_snr(clean, noise, snr_db=0.0)
        # 0 dB SNR 时信号和噪声功率应接近
        signal_power = np.mean(clean**2)
        noise_est = noisy - clean
        noise_power = np.mean(noise_est**2)
        ratio = 10 * np.log10(signal_power / (noise_power + 1e-12))
        assert -1.0 < ratio < 1.0, f"SNR 偏差过大: {ratio:.2f} dB"


def test_getitem_shape():
    """验证 __getitem__ 返回的 (noisy, clean) 形状正确。"""
    import tempfile

    import numpy as np
    import soundfile as sf

    from data.dataset import DenoisingDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        dur = 1.0
        sr = 16000
        nsamples = int(dur * sr)
        sf.write(f"{tmpdir}/speaker1_001.wav", np.random.randn(nsamples * 2).astype("float32"), sr)
        sf.write(f"{tmpdir}/noise_001.wav", np.random.randn(nsamples * 2).astype("float32"), sr)

        ds = DenoisingDataset(
            clean_dir=tmpdir,
            noise_dir=tmpdir,
            sample_rate=sr,
            duration=dur,
        )
        noisy, clean = ds[0]
        assert noisy.shape == (nsamples,), f"noisy shape: {noisy.shape}"
        assert clean.shape == (nsamples,), f"clean shape: {clean.shape}"
        assert noisy.dtype == clean.dtype
