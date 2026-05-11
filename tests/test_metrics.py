"""
tests/test_metrics.py — 评估指标单元测试

验证各指标在极端场景下的行为 (相同信号 / 零信号 / 随机信号)。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_snr_perfect():
    """两段相同信号 SNR 应为 inf。"""
    import numpy as np

    from evaluation.metrics import _compute_snr

    sig = np.random.randn(16000).astype(np.float32)
    snr = _compute_snr(sig, sig)
    assert np.isinf(snr) or snr > 100, f"相同信号 SNR 应很高: {snr}"


def test_sisdr_perfect():
    """两段相同信号 SI-SDR 应为 inf。"""
    import numpy as np

    from evaluation.metrics import _compute_sisdr

    sig = np.random.randn(16000).astype(np.float32)
    sisdr = _compute_sisdr(sig, sig)
    assert np.isinf(sisdr) or sisdr > 100, f"相同信号 SI-SDR 应很高: {sisdr}"


def test_lsd_perfect():
    """两段相同信号 LSD 应接近 0。"""
    import numpy as np

    from evaluation.metrics import _compute_lsd

    np.random.seed(42)
    sig = np.random.randn(16000).astype(np.float32)
    lsd = _compute_lsd(sig, sig)
    assert lsd < 1.0, f"相同信号 LSD 应接近 0: {lsd}"


def test_segsnr_range():
    """验证 SegSNR 在合理范围内。"""
    import numpy as np

    from evaluation.metrics import _compute_segsnr

    sr = 16000
    clean = np.random.randn(sr * 2).astype(np.float32)
    noise = np.random.randn(sr * 2).astype(np.float32) * 0.1
    noisy = clean + noise
    segsnr = _compute_segsnr(clean, noisy, sr)
    # 信号远大于噪声时 SegSNR 应为正
    assert segsnr > 0, f"SegSNR 应为正: {segsnr}"
